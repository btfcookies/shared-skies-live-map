"""HMS smoke: today's file (even if empty), else yesterday's as a flagged
proxy, never anything older."""
import asyncio
import io
import json
import zipfile
from datetime import datetime, timedelta, timezone

import httpx
import shapefile

import backend.main as main
from backend.state import EPOCH
from backend import config, hms, hms_feed

NOAA_HOST = "satepsanone.nesdis.noaa.gov"

TX_SQUARE = [[-98.0, 30.0], [-98.0, 31.0], [-97.0, 31.0], [-97.0, 30.0], [-98.0, 30.0]]
ONTARIO_SQUARE = [[-81.0, 48.0], [-81.0, 49.0], [-80.0, 49.0], [-80.0, 48.0], [-81.0, 48.0]]


def _hms_zip(polygons):
    """A shapefile zip in NOAA's HMS schema; polygons = [(ring, density)]."""
    shp, shx, dbf = io.BytesIO(), io.BytesIO(), io.BytesIO()
    w = shapefile.Writer(shp=shp, shx=shx, dbf=dbf, shapeType=shapefile.POLYGON)
    for name, size in (("Satellite", 20), ("Start", 12), ("End", 12), ("Density", 7)):
        w.field(name, "C", size)
    for ring, density in polygons:
        w.poly([ring])
        w.record("GOES-EAST", "2026294 1200", "2026294 2000", density)
    w.close()
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        for ext, buf in (("shp", shp), ("shx", shx), ("dbf", dbf)):
            zf.writestr(f"hms_smoke.{ext}", buf.getvalue())
    return out.getvalue()


def _route_files(net, files):
    """Serve {date: zip bytes} at NOAA's URL layout; anything else is 404."""
    def handler(request):
        for d, body in files.items():
            if request.url.path.endswith(f"/{d:%Y}/{d:%m}/hms_smoke{d:%Y%m%d}.zip"):
                return httpx.Response(200, content=body)
        return httpx.Response(404)
    net.route(NOAA_HOST, handler)


def _today():
    return datetime.now(timezone.utc).date()


def _requested_days(net):
    return sorted(r.url.path.rsplit("hms_smoke", 1)[1][:8] for r in net.calls(NOAA_HOST))


def test_today_empty_file_means_no_smoke(net):
    today = _today()
    _route_files(net, {today: _hms_zip([(ONTARIO_SQUARE, "Heavy")]),
                       today - timedelta(days=1): _hms_zip([(TX_SQUARE, "Heavy")])})
    fc = asyncio.run(hms.fetch_latest_hms())
    assert fc["data_date"] == today.isoformat()
    assert fc["is_proxy"] is False
    assert fc["count"] == 0 and fc["features"] == []
    # No walk-back once today's file exists.
    assert _requested_days(net) == [f"{today:%Y%m%d}"]


def test_today_missing_uses_yesterday_as_proxy(net):
    today = _today()
    yesterday = today - timedelta(days=1)
    _route_files(net, {yesterday: _hms_zip([(TX_SQUARE, "Heavy"), (ONTARIO_SQUARE, "Light")])})
    fc = asyncio.run(hms.fetch_latest_hms())
    assert fc["data_date"] == yesterday.isoformat()
    assert fc["is_proxy"] is True
    assert fc["count"] == 1 and fc["density_counts"] == {"Heavy": 1}

    polys = hms.build_density_polygons(fc["features"])
    assert hms.smoke_density_at(polys, -97.5, 30.5) == 3
    assert hms.smoke_density_at(polys, -95.0, 29.7) == 0


def test_never_older_than_yesterday(net):
    today = _today()
    _route_files(net, {today - timedelta(days=2): _hms_zip([(TX_SQUARE, "Heavy")]),
                       today - timedelta(days=3): _hms_zip([(TX_SQUARE, "Heavy")])})
    fc = asyncio.run(hms.fetch_latest_hms())
    assert fc["data_date"] is None and fc["count"] == 0 and fc["is_proxy"] is False
    assert _requested_days(net) == sorted([f"{today:%Y%m%d}",
                                           f"{today - timedelta(days=1):%Y%m%d}"])


def test_missing_vs_empty_at_file_level(net):
    today = _today()
    _route_files(net, {today: _hms_zip([(ONTARIO_SQUARE, "Medium")])})
    assert asyncio.run(hms.fetch_hms_for_date(today)) == []
    assert asyncio.run(hms.fetch_hms_for_date(today - timedelta(days=1))) is None


def test_hms_currency():
    now = datetime(2026, 10, 21, 3, 0, tzinfo=timezone.utc)
    assert hms.hms_currency({"data_date": "2026-10-21"}, now) == (True, False)
    assert hms.hms_currency({"data_date": "2026-10-20"}, now) == (True, True)
    assert hms.hms_currency({"data_date": "2026-10-19"}, now) == (False, False)
    assert hms.hms_currency({"data_date": "2026-06-19"}, now) == (False, False)
    assert hms.hms_currency({"data_date": None}, now) == (False, False)
    assert hms.hms_currency(None, now) == (False, False)


def test_refresh_keeps_current_day_when_noaa_has_nothing(monkeypatch, no_snapshots):
    today = _today().isoformat()
    current = {"type": "FeatureCollection", "features": [], "data_date": today,
               "is_proxy": False, "count": 0, "fetched_at": datetime.now(timezone.utc).isoformat()}
    far = datetime.now(timezone.utc) + timedelta(days=1)
    monkeypatch.setattr(hms_feed, "hms_cache", {"data": current, "expires": far})

    async def nothing():
        return {"type": "FeatureCollection", "features": [], "data_date": None,
                "is_proxy": False, "count": 0}

    monkeypatch.setattr(hms, "fetch_latest_hms", nothing)
    asyncio.run(hms_feed._fetch_hms())
    assert hms_feed.hms_cache["data"] is current


def test_hydrate_ignores_stale_hms_snapshot(monkeypatch, tmp_path):
    # Only the HMS snapshot exists; every other path points at nothing.
    for attr in ("TEXAS_SNAPSHOT_PATH", "QUANTUM_SNAPSHOT_PATH", "PURPLEAIR_SNAPSHOT_PATH",
                 "WEATHER_GRID_SNAPSHOT_PATH", "AQ_SNAPSHOT_PATH", "CONDITIONS_SNAPSHOT_PATH"):
        monkeypatch.setattr(config, attr, str(tmp_path / f"{attr}.json"))
    path = tmp_path / "hms.json"
    monkeypatch.setattr(config, "HMS_SNAPSHOT_PATH", str(path))
    fetched = datetime.now(timezone.utc).isoformat()

    for data_date, loaded in (("2026-06-19", False),
                              ((_today() - timedelta(days=1)).isoformat(), True)):
        path.write_text(json.dumps({"type": "FeatureCollection", "features": [], "count": 0,
                                    "data_date": data_date, "fetched_at": fetched}))
        monkeypatch.setattr(hms_feed, "hms_cache", {"data": None, "expires": EPOCH})
        main._hydrate_from_snapshots()
        assert (hms_feed.hms_cache["data"] is not None) is loaded, data_date
