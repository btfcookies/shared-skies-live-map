"""Visit counter (Upstash Redis, SQLite fallback) and the /api/visit, /api/metrics routes."""

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone
import os

from fastapi import APIRouter, Request

from backend import limits
from backend.config import VISITS_DB
from backend.http_client import shared_http
from backend.ratelimit import client_ip

router = APIRouter()


# Upstash Redis REST — set these env vars in Render for persistent visit counts.
# If not set, falls back to SQLite (resets on every server restart).
UPSTASH_URL   = os.environ.get("UPSTASH_REDIS_REST_URL", "").rstrip("/")
UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")


# How long the *publicly-shown* visit count is cached before refreshing.
# This only governs what users see, so the number doesn't obviously tick up
# when someone refreshes.
PUBLIC_VISITS_TTL_MIN = 10


# Module-level cache for the displayed visit count
_public_visits_cache: dict = {"count": None, "expires": datetime.min.replace(tzinfo=timezone.utc)}


# ── Persistent visits counter helpers ────────────────────────────────────────
# Priority: Upstash Redis (truly persistent across restarts) > SQLite (ephemeral)


def init_db() -> None:
    """Create the SQLite visit counter (local dev, and the fallback when
    Upstash is not configured)."""
    conn = sqlite3.connect(VISITS_DB, timeout=5)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS visits (id INTEGER PRIMARY KEY CHECK (id = 1), count INTEGER)")
    cur.execute("INSERT OR IGNORE INTO visits (id, count) VALUES (1, 0)")
    conn.commit()
    conn.close()


def _get_visit_count_sync():
    conn = sqlite3.connect(VISITS_DB, timeout=5)
    cur = conn.cursor()
    cur.execute("SELECT count FROM visits WHERE id=1")
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else 0


def _inc_visit_sync():
    conn = sqlite3.connect(VISITS_DB, timeout=5)
    cur = conn.cursor()
    cur.execute("UPDATE visits SET count = count + 1 WHERE id = 1")
    conn.commit()
    cur.execute("SELECT count FROM visits WHERE id=1")
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else 0


async def _upstash_incr() -> int:
    """Atomically increment the visit counter in Upstash Redis and return the new value."""
    resp = await shared_http().post(
        f"{UPSTASH_URL}/incr/shared_skies_visits",
        headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
        timeout=5.0,
    )
    resp.raise_for_status()
    return int(resp.json()["result"])


async def _upstash_get() -> int:
    """Get the current visit count from Upstash Redis."""
    resp = await shared_http().get(
        f"{UPSTASH_URL}/get/shared_skies_visits",
        headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
        timeout=5.0,
    )
    resp.raise_for_status()
    val = resp.json().get("result")
    return int(val) if val is not None else 0


async def _get_public_visit_count() -> int:
    """Return the 10-min-cached visit count shown to users.
    The stored count moves on every counted /api/visit; this layer only
    controls what's displayed so the number can't be watched ticking up.
    All clients in the same 10-min window see the same value."""
    global _public_visits_cache
    now = datetime.now(timezone.utc)
    cached_count = _public_visits_cache.get("count")
    expires_at = _public_visits_cache.get("expires")

    if cached_count is not None and expires_at is not None and now < expires_at:
        return cached_count

    # Cache miss or expired — read the true count and refresh the cache.
    try:
        if UPSTASH_URL and UPSTASH_TOKEN:
            count = await _upstash_get()
        else:
            count = await asyncio.to_thread(_get_visit_count_sync)
    except Exception as e:
        print(f"Public visit cache refresh failed: {e}")
        # If reading fails, keep showing the last good value rather than zero.
        return cached_count if cached_count is not None else 0

    _public_visits_cache = {
        "count": count,
        "expires": now + timedelta(minutes=PUBLIC_VISITS_TTL_MIN),
    }
    return count


@router.post("/api/visit")
async def record_visit(request: Request):
    """Count a visit at most once per client IP per VISIT_DEDUPE_MIN, then
    return the *publicly-cached* total so users can't watch the number tick up
    by refreshing."""
    if limits.visit_dedupe.first_time(client_ip(request)) and limits.visit_budget.check(limits.GLOBAL_KEY)[0]:
        try:
            if UPSTASH_URL and UPSTASH_TOKEN:
                await _upstash_incr()
            else:
                await asyncio.to_thread(_inc_visit_sync)
        except Exception as e:
            # With Upstash configured, a local SQLite write would never be read
            # back (the public count reads Upstash), so the visit is dropped and
            # logged rather than silently diverted.
            print(f"Visit increment failed: {type(e).__name__}: {e}")
    # Return the display-cached count (refreshes every PUBLIC_VISITS_TTL_MIN)
    try:
        return {"visits": await _get_public_visit_count()}
    except Exception:
        return {"visits": 0}


@router.get("/api/metrics")
async def get_metrics():
    """Return the publicly-cached visit count (refreshes every ~10 min).
    For the raw real-time count, read Upstash/SQLite directly."""
    try:
        return {"visits": await _get_public_visit_count()}
    except Exception as e:
        print(f"Metrics read error: {e}")
        return {"visits": 0}
