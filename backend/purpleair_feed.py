"""Live PurpleAir sensor readings: cache, refresh and freshness checks."""

import os
from datetime import datetime, timedelta, timezone

from backend import config, snapshots
from backend.config import PURPLEAIR_BACKOFF_MIN, PURPLEAIR_CACHE_TTL_MIN, PURPLEAIR_MAX_AGE_MIN
from backend.state import EPOCH
from backend.util import age_min, parse_ts, single_flight


# Live PurpleAir sensor cache (list of {lat, lon, pm25} for current TX air).
# next_attempt drives the refresh loop's schedule and failure backoff.
purpleair_cache: dict = {"data": None, "next_attempt": EPOCH, "failures": 0}


def hydrate(now: datetime) -> None:
    """Live PurpleAir readings from the last snapshot, with their real age."""
    global purpleair_cache
    # PurpleAir: loaded with its real fetch time. live_purpleair() refuses it
    # once it is older than PURPLEAIR_MAX_AGE_MIN, and the loop skips the
    # startup fetch only while it is younger than the TTL, so a dev server on
    # --reload doesn't bill a full call per file save.
    pa_snap = snapshots.load(config.PURPLEAIR_SNAPSHOT_PATH)
    p_ts = parse_ts((pa_snap or {}).get("fetched_at"))
    if pa_snap and pa_snap.get("usable") and p_ts:
        purpleair_cache = {"data": pa_snap, "failures": 0,
                           "next_attempt": p_ts + timedelta(minutes=PURPLEAIR_CACHE_TTL_MIN)}
        age = age_min(pa_snap["fetched_at"], now)
        print(f"Loaded PurpleAir snapshot ({pa_snap.get('count', 0)} live sensors, "
              f"{age:.0f} min old" + ("; too old to use as live)" if age > PURPLEAIR_MAX_AGE_MIN else ")"))


def purpleair_key_configured() -> bool:
    return bool(os.environ.get("PURPLEAIR_API_KEY", "").strip())


def live_purpleair(now: datetime | None = None) -> dict | None:
    """The cached PurpleAir snapshot when it is usable and young enough to
    count as live (PURPLEAIR_MAX_AGE_MIN), else None (climatology fallback)."""
    snap = purpleair_cache.get("data")
    if not snap or not snap.get("usable"):
        return None
    age = age_min(snap.get("fetched_at"), now)
    if age is None or age > PURPLEAIR_MAX_AGE_MIN:
        return None
    return snap


async def _fetch_purpleair() -> bool:
    """One statewide PurpleAir fetch. Schedules the next attempt: the TTL on
    success, a doubling backoff (5 min up to the TTL) on failure. A failed
    fetch never wipes a good prior snapshot. Returns True on success."""
    global purpleair_cache
    from backend.purpleair import fetch_live_snapshot

    try:
        snap = await fetch_live_snapshot()
    except Exception as e:
        # An unexpected parse error must take the same backoff as an HTTP
        # error; otherwise next_attempt stays in the past and the loop bills
        # a full call about once a minute.
        print(f"[purpleair] snapshot raised: {type(e).__name__}: {e}")
        snap = {"usable": False, "count": 0, "error": type(e).__name__}
    now = datetime.now(timezone.utc)
    if snap.get("usable"):
        purpleair_cache = {
            "data": snap,
            "failures": 0,
            "next_attempt": now + timedelta(minutes=PURPLEAIR_CACHE_TTL_MIN),
        }
        await snapshots.save_async(config.PURPLEAIR_SNAPSHOT_PATH, snap)
        print(f"✓ PurpleAir refreshed: {snap['count']} live sensors, "
              f"statewide mean={snap.get('statewide_mean'):.2f}")
        return True
    failures = purpleair_cache.get("failures", 0) + 1
    backoff = min(PURPLEAIR_BACKOFF_MIN * 2 ** (failures - 1), PURPLEAIR_CACHE_TTL_MIN)
    purpleair_cache["failures"] = failures
    purpleair_cache["next_attempt"] = now + timedelta(minutes=backoff)
    print(f"PurpleAir refresh failed ({snap.get('error') or 'too few sensors'}, "
          f"{snap.get('count', 0)} usable sensors); keeping prior data, "
          f"retry in {backoff} min")
    return False


async def refresh_purpleair() -> bool:
    return await single_flight("purpleair", _fetch_purpleair)
