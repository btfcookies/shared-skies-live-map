"""Small helpers: single-flight tasks, timestamp parsing, process memory."""

import asyncio
import ctypes
import os
import sys
from datetime import datetime, timezone


# Single-flight: one shared asyncio.Task per resource. Every caller (the
# refresh loop, a cold-start request, the quantum solve) awaits the same task,
# so two computes of the same thing can never overlap on the 0.1-CPU instance.
_inflight: dict[str, asyncio.Task] = {}


def _consume_task_exception(task: asyncio.Task) -> None:
    # Callers that were cancelled (client disconnect) leave the shared task's
    # exception unobserved; read it here so asyncio doesn't log it as lost.
    if not task.cancelled():
        task.exception()


async def single_flight(key: str, factory):
    """Run factory() as the one in-flight task for `key`, or join the task
    already running. shield() keeps a cancelled caller from cancelling the
    work other callers are waiting on."""
    task = _inflight.get(key)
    if task is None or task.done():
        task = asyncio.create_task(factory(), name=f"single-flight:{key}")
        task.add_done_callback(_consume_task_exception)
        _inflight[key] = task
    return await asyncio.shield(task)


def in_flight(key: str) -> bool:
    task = _inflight.get(key)
    return task is not None and not task.done()


def retry_after(due: datetime, now: datetime) -> str:
    """Retry-After header value (whole seconds, at least 1) until `due`."""
    return str(max(1, int((due - now).total_seconds() + 0.999)))


def parse_ts(raw) -> datetime | None:
    """ISO timestamp -> aware UTC datetime, or None."""
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def age_min(raw, now: datetime | None = None) -> float | None:
    ts = parse_ts(raw)
    if ts is None:
        return None
    return ((now or datetime.now(timezone.utc)) - ts).total_seconds() / 60.0


def malloc_trim() -> None:
    """Hand freed heap pages back to the OS (glibc only). After the model load
    and after each compute this is the difference between RSS settling and
    RSS ratcheting up toward the 512 MB limit."""
    if not sys.platform.startswith("linux"):
        return
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def rss_mb() -> float | None:
    """Current resident set size, cheaply (no subprocess, no compute)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return round(pages * os.sysconf("SC_PAGE_SIZE") / 2**20, 1)
    except Exception:
        pass
    try:
        import psutil  # dev machines only; not a deploy dependency
        return round(psutil.Process().memory_info().rss / 2**20, 1)
    except Exception:
        return None
