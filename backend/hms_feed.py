"""NOAA HMS smoke polygons: cache, refresh and the /api/live/hms-smoke route."""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter

from backend import config, snapshots
from backend.state import EPOCH
from backend.util import in_flight, parse_ts, single_flight


router = APIRouter()


# NOAA HMS smoke polygon cache, refreshed by the background loop.
hms_cache: dict = {"data": None, "expires": EPOCH}


HMS_CACHE_TTL_MIN = 60  # HMS analysts publish daily and re-draw a few times/day


def hydrate(now: datetime) -> None:
    """NOAA smoke polygons from the last snapshot, if still current."""
    global hms_cache
    # HMS: only a snapshot whose data_date is still today or yesterday (UTC).
    # The old hydrate served a June snapshot as live smoke in October.
    from backend.hms import hms_currency
    hms_snap = snapshots.load(config.HMS_SNAPSHOT_PATH)
    h_ts = parse_ts((hms_snap or {}).get("fetched_at"))
    if hms_snap and h_ts and hms_currency(hms_snap, now)[0]:
        hms_cache = {"data": hms_snap, "expires": h_ts + timedelta(minutes=HMS_CACHE_TTL_MIN)}
        print(f"Loaded HMS snapshot ({hms_snap.get('count', 0)} polygons, "
              f"data_date={hms_snap.get('data_date')})")
    elif hms_snap:
        print(f"Ignoring HMS snapshot (data_date={hms_snap.get('data_date')} is not "
              f"today or yesterday UTC)")


async def _fetch_hms() -> None:
    """Fetch today's (or, before it is published, yesterday's) HMS smoke
    polygons from NOAA, clipped to the TX bbox; cache + snapshot. Run
    single-flight by the refresh loop and by a cold-start overlay request."""
    global hms_cache
    from backend.hms import fetch_latest_hms, hms_currency
    now = datetime.now(timezone.utc)
    try:
        result = await fetch_latest_hms()
    except Exception as e:
        print(f"HMS refresh failed: {type(e).__name__}: {e}")
        hms_cache["expires"] = now + timedelta(minutes=15)
        return
    old = hms_cache.get("data")
    old_usable = hms_currency(old, now)[0]
    if old_usable and (result.get("data_date") is None
                       or result["data_date"] < old["data_date"]):
        # NOAA unreachable (or today's file vanished mid-outage): a still-
        # current cached day beats an older or empty answer.
        print(f"HMS refresh found nothing newer than {old['data_date']}; keeping it")
        hms_cache["expires"] = now + timedelta(minutes=15)
        return
    if result.get("data_date") is None:
        # Neither today's nor yesterday's file could be read (NOAA down, or not
        # published). That is a failure, not a "no smoke" day: retry in 15 min
        # instead of caching an empty day for the full TTL.
        print("HMS refresh failed: no file for today or yesterday; retry in 15 min")
        hms_cache["expires"] = now + timedelta(minutes=15)
        return
    hms_cache = {"data": result, "expires": now + timedelta(minutes=HMS_CACHE_TTL_MIN)}
    await snapshots.save_async(config.HMS_SNAPSHOT_PATH, result)
    print(f"✓ HMS refreshed: {result.get('count', 0)} polygons "
          f"(data_date={result.get('data_date')}"
          f"{', proxy for today' if result.get('is_proxy') else ''})")


async def refresh_hms() -> None:
    await single_flight("hms", _fetch_hms)


@router.get("/api/live/hms-smoke")
async def hms_smoke():
    """Returns today's NOAA HMS smoke polygons clipped to Texas, as a GeoJSON
    FeatureCollection, with data_date and is_proxy (true when yesterday's
    file stands in because today's is not published yet). Smoke older than
    yesterday is never served."""
    from backend.hms import hms_currency
    now = datetime.now(timezone.utc)
    try:
        # Before the first good fetch, join or start one, but only once the
        # failure backoff has passed; the refresh loop retries on its own.
        if hms_cache.get("data") is None and (in_flight("hms") or now >= hms_cache["expires"]):
            await refresh_hms()
        data = hms_cache.get("data")
        usable, is_proxy = hms_currency(data, now)
        if usable:
            return {**data, "is_proxy": is_proxy}
        error = None
    except Exception as e:
        print(f"HMS endpoint error: {type(e).__name__}: {e}")
        error = "unavailable"
    # Don't 500 the frontend just because NOAA is down — return empty FC.
    empty = {
        "type": "FeatureCollection",
        "features": [],
        "fetched_at": now.isoformat(),
        "data_date": None,
        "is_proxy": False,
        "count": 0,
        "density_counts": {},
    }
    if error:
        empty["error"] = error
    return empty
