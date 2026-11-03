"""Disk snapshots of cached API responses, so a restart on a persistent disk serves instantly."""

import asyncio
import json
import os


def save(path: str, data: dict | bytes) -> None:
    """Write a cached API response (dict, or already-encoded JSON bytes) to
    disk so a restart on a persistent disk can serve it instantly. Blocking:
    callers on the event loop go through save_async."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        if isinstance(data, bytes):
            with open(tmp, "wb") as f:
                f.write(data)
        else:
            with open(tmp, "w") as f:
                json.dump(data, f, default=str)
        os.replace(tmp, path)
    except Exception as e:
        print(f"Snapshot save failed ({path}): {e}")


async def save_async(path: str, data: dict | bytes) -> None:
    await asyncio.to_thread(save, path, data)


def load(path: str) -> dict | None:
    """Load a previously persisted API response (returns None if missing/corrupt)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Snapshot load failed ({path}): {e}")
        return None
