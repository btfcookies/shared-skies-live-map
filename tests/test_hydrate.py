"""Startup hydration: each data feed restores its own cache from its disk
snapshot, keeping the snapshot's real age (never presenting old data as fresh).

Each feed owns its cache and rebinds it with ``global`` inside its own
``hydrate()``, so these tests fail if a feed stops restoring its cache. The
HMS path has its own test in test_hms.py.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

import backend.main as main
from backend import config, hms_feed, placement, purpleair_feed, weather_feed
from backend.state import EPOCH

SNAPSHOT_PATHS = ("TEXAS_SNAPSHOT_PATH", "QUANTUM_SNAPSHOT_PATH", "HMS_SNAPSHOT_PATH",
                  "PURPLEAIR_SNAPSHOT_PATH", "WEATHER_GRID_SNAPSHOT_PATH", "AQ_SNAPSHOT_PATH",
                  "CONDITIONS_SNAPSHOT_PATH")
STATE_KEYS = ("cache_texas", "cache_texas_by_geoid", "texas_payload", "city_cache")


def _iso(**ago):
    return (datetime.now(timezone.utc) - timedelta(**ago)).isoformat()


def _cold(**extra):
    return {"data": None, "expires": EPOCH, **extra}


@pytest.fixture
def snaps(monkeypatch, tmp_path):
    """Every snapshot path points into an empty temp dir, and every cache and
    Texas state slot starts cold. monkeypatch restores all of it afterwards,
    including caches a hydrate() rebinds."""
    for attr in SNAPSHOT_PATHS:
        monkeypatch.setattr(config, attr, str(tmp_path / f"{attr}.json"))
    monkeypatch.setattr(hms_feed, "hms_cache", _cold())
    monkeypatch.setattr(placement, "quantum_cache", _cold())
    monkeypatch.setattr(purpleair_feed, "purpleair_cache", _cold(next_attempt=EPOCH, failures=0))
    monkeypatch.setattr(weather_feed, "_wgrid_cache", _cold())
    monkeypatch.setattr(weather_feed, "_aq_cache", _cold())
    monkeypatch.setattr(weather_feed, "conditions_cache", _cold())
    for key in STATE_KEYS:
        monkeypatch.setitem(main.state, key, "untouched")

    def write(attr, data):
        (tmp_path / f"{attr}.json").write_text(json.dumps(data))
    return write


def test_no_snapshots_leaves_everything_cold(snaps):
    main._hydrate_from_snapshots()
    assert hms_feed.hms_cache["data"] is None
    assert placement.quantum_cache["data"] is None
    assert purpleair_feed.purpleair_cache["data"] is None
    assert weather_feed._wgrid_cache["data"] is None
    assert weather_feed._aq_cache["data"] is None
    assert weather_feed.conditions_cache["data"] is None
    assert all(main.state[k] == "untouched" for k in STATE_KEYS)


def test_quantum_snapshot_keeps_its_real_age(snaps):
    generated = _iso(minutes=10)
    snaps("QUANTUM_SNAPSHOT_PATH", {"generated_at": generated, "sites": []})
    main._hydrate_from_snapshots()
    cache = placement.quantum_cache
    assert cache["data"]["generated_at"] == generated
    assert cache["expires"] == (datetime.fromisoformat(generated)
                                + timedelta(minutes=placement.QUANTUM_CACHE_TTL_MIN))


def test_purpleair_snapshot_restored_with_fetch_time(snaps):
    fetched = _iso(minutes=30)
    snaps("PURPLEAIR_SNAPSHOT_PATH", {"fetched_at": fetched, "usable": True, "count": 2,
                                      "sensors": []})
    main._hydrate_from_snapshots()
    cache = purpleair_feed.purpleair_cache
    assert cache["data"]["fetched_at"] == fetched
    assert cache["failures"] == 0
    # The schedule continues from the real fetch time, so a restart does not
    # spend a fresh PurpleAir pull.
    assert cache["next_attempt"] == (datetime.fromisoformat(fetched)
                                     + timedelta(minutes=config.PURPLEAIR_CACHE_TTL_MIN))


def test_unusable_purpleair_snapshot_is_ignored(snaps):
    snaps("PURPLEAIR_SNAPSHOT_PATH", {"fetched_at": _iso(minutes=5), "usable": False})
    main._hydrate_from_snapshots()
    assert purpleair_feed.purpleair_cache["data"] is None


def test_weather_snapshots_restore_each_grid(snaps):
    fetched = _iso(minutes=5)
    for attr in ("WEATHER_GRID_SNAPSHOT_PATH", "AQ_SNAPSHOT_PATH", "CONDITIONS_SNAPSHOT_PATH"):
        snaps(attr, {"fetched_at": fetched, "n_cells": 7})
    main._hydrate_from_snapshots()
    assert weather_feed._wgrid_cache["data"]["n_cells"] == 7
    assert weather_feed._aq_cache["data"]["n_cells"] == 7
    assert weather_feed.conditions_cache["data"]["n_cells"] == 7
    assert weather_feed.conditions_cache["expires"] == (
        datetime.fromisoformat(fetched) + timedelta(minutes=weather_feed.CONDITIONS_TTL_MIN))


def test_weather_snapshot_without_timestamp_is_expired_but_kept(snaps):
    # Stale data is still the fallback if the upstream quota runs out.
    snaps("WEATHER_GRID_SNAPSHOT_PATH", {"n_cells": 3})
    main._hydrate_from_snapshots()
    assert weather_feed._wgrid_cache["data"]["n_cells"] == 3
    assert weather_feed._wgrid_cache["expires"] == EPOCH


def test_texas_snapshot_installed_with_index_and_payload(snaps):
    generated = _iso(minutes=5)
    snaps("TEXAS_SNAPSHOT_PATH", {"generated_at": generated,
                                  "tracts": [{"geoid": "48113000100", "pm25": 7.1}]})
    main._hydrate_from_snapshots()
    assert main.state["cache_texas"]["data"]["generated_at"] == generated
    assert main.state["cache_texas_by_geoid"]["48113000100"]["pm25"] == 7.1
    payload = main.state["texas_payload"]
    assert payload["etag"].startswith('"') and payload["raw"] and payload["gz"]
    assert main.state["city_cache"] == {}


def test_old_texas_snapshot_is_not_served(snaps):
    snaps("TEXAS_SNAPSHOT_PATH", {"generated_at": _iso(hours=25),
                                  "tracts": [{"geoid": "48113000100", "pm25": 7.1}]})
    main._hydrate_from_snapshots()
    assert main.state["cache_texas"] == "untouched"


def test_one_corrupt_snapshot_does_not_block_the_rest(snaps, tmp_path):
    # snapshots.load() returns None for unreadable files, so hydration of the
    # other feeds still runs.
    (tmp_path / "TEXAS_SNAPSHOT_PATH.json").write_text("{not json")
    snaps("QUANTUM_SNAPSHOT_PATH", {"generated_at": _iso(minutes=1), "sites": []})
    main._hydrate_from_snapshots()
    assert main.state["cache_texas"] == "untouched"
    assert placement.quantum_cache["data"] is not None
