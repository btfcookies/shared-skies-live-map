"""PurpleAir: QC of the live pull, no calls without a key, refresh backoff and
the age limit on "live" data. All offline; the key used here is a dummy."""
import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from backend.state import EPOCH
from backend import config, purpleair as pa, purpleair_feed

PA_HOST = "api.purpleair.com"
DUMMY_KEY = "test-key-not-real"


def test_no_key_means_no_request(net, monkeypatch):
    monkeypatch.delenv("PURPLEAIR_API_KEY", raising=False)
    assert asyncio.run(pa.fetch_live_sensors()) == []
    snap = asyncio.run(pa.fetch_live_snapshot())
    assert snap["usable"] is False and snap["error"] == "no key"
    assert net.requests == []


def test_live_pull_qc(net):
    fields = ["sensor_index", "latitude", "longitude", "pm2.5_24hour", "pm2.5_atm", "confidence"]
    rows = [
        [1, 30.1, -97.1, 8.0, 3326.0, 100],    # 24-h mean wins over a glitchy atm
        [2, 30.2, -97.2, None, 6.5, 90],       # no 24-h mean: fall back to atm
        [3, 30.3, -97.3, 7.0, 7.0, 40],        # low A/B confidence
        [4, 30.4, -97.4, 250.0, 250.0, 100],   # above the training cap
        [5, None, -97.5, 7.0, 7.0, 100],       # no location
        [6, 30.6, -97.6, -1.0, 2.0, 100],      # negative
    ]
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"fields": fields, "data": rows})

    net.route(PA_HOST, handler)
    sensors, status, error = asyncio.run(pa._fetch_live(DUMMY_KEY))
    assert (status, error) == (200, None)
    assert sensors == [{"lat": 30.1, "lon": -97.1, "pm25": 8.0},
                       {"lat": 30.2, "lon": -97.2, "pm25": 6.5}]

    req = seen[0]
    assert req.headers["X-API-Key"] == DUMMY_KEY
    assert req.url.params["location_type"] == "0"
    assert req.url.params["max_age"] == str(pa.MAX_AGE_SEC)
    assert "pm2.5_24hour" in req.url.params["fields"].split(",")


@pytest.mark.parametrize("code", [401, 402, 403, 500])
def test_live_pull_http_errors(net, code):
    net.route(PA_HOST, lambda request: httpx.Response(code, text="nope"))
    sensors, status, error = asyncio.run(pa._fetch_live(DUMMY_KEY))
    assert sensors == [] and status == code and error == f"HTTP {code}"


def _snap(usable, fetched_at=None, n=30):
    sensors = [{"lat": 30 + i * 0.01, "lon": -97.0, "pm25": 5.0} for i in range(n)] if usable else []
    return {"sensors": sensors, "count": len(sensors), "usable": usable,
            "statewide_mean": 5.0 if usable else None, "http_status": 200 if usable else 402,
            "error": None if usable else "HTTP 402",
            "fetched_at": (fetched_at or datetime.now(timezone.utc)).isoformat()}


def test_refresh_backoff_and_recovery(monkeypatch, no_snapshots):
    monkeypatch.setattr(purpleair_feed, "purpleair_cache",
                        {"data": None, "next_attempt": EPOCH, "failures": 0})

    async def failing():
        return _snap(False)

    async def working():
        return _snap(True)

    def scheduled_in_min():
        delta = purpleair_feed.purpleair_cache["next_attempt"] - datetime.now(timezone.utc)
        return round(delta.total_seconds() / 60)

    monkeypatch.setattr(pa, "fetch_live_snapshot", failing)
    gaps = []
    for _ in range(4):
        assert asyncio.run(purpleair_feed._fetch_purpleair()) is False
        gaps.append(scheduled_in_min())
    assert gaps == [5, 10, 15, 15]

    monkeypatch.setattr(pa, "fetch_live_snapshot", working)
    assert asyncio.run(purpleair_feed._fetch_purpleair()) is True
    assert purpleair_feed.purpleair_cache["failures"] == 0
    assert scheduled_in_min() == config.PURPLEAIR_CACHE_TTL_MIN == 15

    # A later failure keeps the good snapshot.
    monkeypatch.setattr(pa, "fetch_live_snapshot", failing)
    asyncio.run(purpleair_feed._fetch_purpleair())
    assert purpleair_feed.purpleair_cache["data"]["usable"] is True


def test_live_data_age_limit(monkeypatch):
    now = datetime.now(timezone.utc)
    fresh = _snap(True, now - timedelta(minutes=config.PURPLEAIR_MAX_AGE_MIN - 1))
    stale = _snap(True, now - timedelta(minutes=config.PURPLEAIR_MAX_AGE_MIN + 1))
    monkeypatch.setattr(purpleair_feed, "purpleair_cache", {"data": fresh, "next_attempt": now, "failures": 0})
    assert purpleair_feed.live_purpleair(now) is fresh
    monkeypatch.setattr(purpleair_feed, "purpleair_cache", {"data": stale, "next_attempt": now, "failures": 0})
    assert purpleair_feed.live_purpleair(now) is None
    monkeypatch.setattr(purpleair_feed, "purpleair_cache",
                        {"data": _snap(False), "next_attempt": now, "failures": 0})
    assert purpleair_feed.live_purpleair(now) is None
