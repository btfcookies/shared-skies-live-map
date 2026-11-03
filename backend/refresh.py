"""Background refresh loop: PurpleAir, HMS, predictions and sensor placement on their schedules."""

import asyncio
from datetime import datetime, timezone

from backend import hms_feed, placement, purpleair_feed
from backend.hms_feed import refresh_hms
from backend.predictions import refresh_texas
from backend.purpleair_feed import purpleair_key_configured, refresh_purpleair
from backend.state import EPOCH, state
from backend.util import age_min, in_flight, single_flight


# ── Background refresh loop ──────────────────────────────────────────────────
# ONE loop owns every scheduled refresh, independent of traffic: requests only
# read caches (or, before the first result exists, wait for it). Render's free
# instance stays up through the keepalive, so the map stays current even when
# nobody is looking, and PurpleAir spend is a fixed schedule rather than a
# function of traffic.


async def _ensure_quantum() -> None:
    # Before expires there is either a current result or a failure backoff.
    if datetime.now(timezone.utc) < placement.quantum_cache["expires"]:
        return
    if not in_flight("quantum"):
        print("Solving sensor placement...")
    try:
        await single_flight("quantum", placement.solve_quantum_and_cache)
    except Exception as e:
        print(f"Sensor placement solve failed: {type(e).__name__}: {e}")


async def _startup_refresh() -> None:
    """Startup order matters: live inputs first, so the first compute is a
    real one rather than a climatology placeholder."""
    now = datetime.now(timezone.utc)
    if purpleair_key_configured():
        if now >= purpleair_feed.purpleair_cache["next_attempt"]:
            print("Startup: fetching live PurpleAir sensors...")
            await refresh_purpleair()
        else:
            age = age_min(purpleair_feed.purpleair_cache["data"]["fetched_at"], now)
            print(f"Startup: PurpleAir snapshot is {age:.0f} min old; skipping the fetch")
    if now >= hms_feed.hms_cache["expires"]:
        print("Startup: fetching NOAA HMS smoke polygons...")
        await refresh_hms()
    if state.get("bundle") is not None and state.get("tract_lookup") is not None:
        print("Startup: computing Texas predictions...")
        try:
            await refresh_texas()
        except Exception as e:
            print(f"Startup Texas compute failed: {type(e).__name__}: {e}")


async def _refresh_cycle() -> None:
    now = datetime.now(timezone.utc)
    pa_refreshed = False
    # Each input step is isolated: one raising must not skip the compute.
    if purpleair_key_configured() and now >= purpleair_feed.purpleair_cache["next_attempt"]:
        try:
            pa_refreshed = await refresh_purpleair()
        except Exception as e:
            print(f"PurpleAir refresh error: {type(e).__name__}: {e}")
    if now >= hms_feed.hms_cache["expires"]:
        try:
            await refresh_hms()
        except Exception as e:
            print(f"HMS refresh error: {type(e).__name__}: {e}")
    if state.get("bundle") is not None and state.get("tract_lookup") is not None:
        if pa_refreshed or now >= state.get("texas_next_due", EPOCH):
            try:
                await refresh_texas()
            except Exception as e:
                print(f"Texas compute failed: {type(e).__name__}: {e}")
    # Lowest priority: only after predictions exist.
    if state["cache_texas"]["data"] is not None:
        await _ensure_quantum()


def _next_wakeup_s() -> float:
    due = [hms_feed.hms_cache["expires"]]
    if purpleair_key_configured():
        due.append(purpleair_feed.purpleair_cache["next_attempt"])
    if state.get("bundle") is not None and state.get("tract_lookup") is not None:
        due.append(state.get("texas_next_due", EPOCH))
    if state["cache_texas"]["data"] is not None:
        due.append(placement.quantum_cache["expires"])
    delay = (min(due) - datetime.now(timezone.utc)).total_seconds()
    # Re-check at least once a minute; never spin.
    return min(max(delay, 1.0), 60.0)


async def refresh_loop() -> None:
    try:
        await _startup_refresh()
    except Exception as e:
        print(f"Startup refresh failed: {type(e).__name__}: {e}")
    finally:
        state["texas_first_attempt"].set()
    if state["cache_texas"]["data"] is not None:
        await _ensure_quantum()
    while True:
        try:
            await asyncio.sleep(_next_wakeup_s())
            await _refresh_cycle()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"Refresh loop error: {type(e).__name__}: {e}")
            await asyncio.sleep(60)
