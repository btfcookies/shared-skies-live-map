"""Small in-process, per-client-IP limits for the public API.

The backend is a single uvicorn process, so plain dicts are enough: no Redis,
no extra dependency. Memory is bounded by evicting the least recently seen
keys once a table reaches its cap, and keys are stored as 8-byte digests, so a
flood of spoofed or rotating IPs fills all three tables in main.py with under
7 MB (measured). The per-IP limits trust X-Forwarded-For, so main.py also keeps
global budgets (a single-key FixedWindowLimiter) on every upstream call a
request can trigger.
"""
import hashlib
import time
from collections import OrderedDict

from fastapi import Request


def client_ip(request: Request) -> str:
    """The caller's IP: the left-most X-Forwarded-For entry if present (set by
    Vercel's rewrite proxy and Render's load balancer), else the socket peer.
    A direct caller can forge the header, so these limits slow abuse down;
    they are not an authentication boundary."""
    fwd = request.headers.get("x-forwarded-for", "")
    first = fwd.split(",")[0].strip()
    if first:
        return first[:64]
    return request.client.host if request.client else "unknown"


def _digest(key: str) -> bytes:
    # A 64-char key string costs ~113 bytes per entry; 8 digest bytes cost 41,
    # and a collision only merges two callers' counts.
    return hashlib.blake2b(key.encode(), digest_size=8).digest()


class FixedWindowLimiter:
    """At most `limit` requests per `window_s` seconds per key."""

    def __init__(self, limit: int, window_s: float = 60.0, max_keys: int = 10_000):
        self.limit = limit
        self.window_s = window_s
        self.max_keys = max_keys
        self._windows: OrderedDict[bytes, list] = OrderedDict()  # key digest -> [start, count]

    def check(self, key: str) -> tuple[bool, int]:
        """(allowed, retry_after_seconds). Counts the request when allowed."""
        key = _digest(key)
        now = time.monotonic()
        win = self._windows.get(key)
        if win is None or now - win[0] >= self.window_s:
            win = [now, 0]
            self._windows[key] = win
        self._windows.move_to_end(key)
        while len(self._windows) > self.max_keys:
            self._windows.popitem(last=False)
        if win[1] >= self.limit:
            return False, max(1, int(win[0] + self.window_s - now + 0.999))
        win[1] += 1
        return True, 0


class RecentKeys:
    """Remembers keys for `ttl_s` seconds: first_time(key) is True at most
    once per key per ttl window."""

    def __init__(self, ttl_s: float, max_keys: int = 10_000):
        self.ttl_s = ttl_s
        self.max_keys = max_keys
        self._seen: OrderedDict[bytes, float] = OrderedDict()  # key digest -> last counted

    def first_time(self, key: str) -> bool:
        key = _digest(key)
        now = time.monotonic()
        # Entries are in insertion order of their last count, so expired ones
        # sit at the front.
        while self._seen:
            ts = next(iter(self._seen.values()))
            if now - ts < self.ttl_s and len(self._seen) <= self.max_keys:
                break
            self._seen.popitem(last=False)
        if key in self._seen:
            return False
        self._seen[key] = now
        return True
