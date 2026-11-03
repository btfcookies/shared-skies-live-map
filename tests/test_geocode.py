"""/api/geocode contract: suggest = Photon only, submit adds the throttled
Nominatim fallback, bounded q, per-IP limit. Upstreams are mocked."""
import itertools

import httpx
import pytest

from backend import geocode, limits
from backend.ratelimit import FixedWindowLimiter

PHOTON = "photon.komoot.io"
NOMINATIM = "nominatim.openstreetmap.org"
_ips = (f"198.51.100.{i}" for i in itertools.count(1))


def _photon_feature(name, lat, lon, state="Texas"):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"name": name, "state": state, "county": "Some County",
                           "osm_value": "city"}}


def _photon(places):
    """Photon answers with `places` for queries naming them, else nothing."""
    def handler(request):
        q = request.url.params["q"].lower()
        feats = [f for key, f in places.items() if key in q]
        return httpx.Response(200, json={"type": "FeatureCollection", "features": feats})
    return handler


NOMINATIM_HIT = [{"display_name": "Old Town Spring, Spring, Texas, USA",
                  "lat": "30.08", "lon": "-95.42", "type": "neighbourhood"}]


@pytest.fixture
def geo(net, monkeypatch):
    """Fresh cache, limiter and Nominatim throttle; a client IP per test."""
    monkeypatch.setattr(geocode, "_geocode_cache", type(geocode._geocode_cache)())
    lim = limits.geocode_limiter
    monkeypatch.setattr(limits, "geocode_limiter", FixedWindowLimiter(lim.limit, lim.window_s))
    monkeypatch.setattr(geocode, "_nominatim_state", {"last": 0.0, "lock": None})
    net.route(PHOTON, _photon({
        "houston": _photon_feature("Houston", 29.76, -95.37),
        "shreveport": _photon_feature("Shreveport", 32.5, -93.75, state="Louisiana"),
    }))
    net.route(NOMINATIM, lambda request: httpx.Response(200, json=NOMINATIM_HIT))
    ip = next(_ips)
    return net, {"X-Forwarded-For": ip}


def test_suggest_is_default_and_photon_only(client, geo):
    net, headers = geo
    r = client.get("/api/geocode", params={"q": "  houston  "}, headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body[0]["name"] == "Houston"
    assert set(body[0]) == {"name", "display_name", "lat", "lon", "type"}
    # A miss while typing never falls back to Nominatim.
    r = client.get("/api/geocode", params={"q": "zzqxw plorb"}, headers=headers)
    assert r.status_code == 200 and r.json() == []
    assert net.calls(NOMINATIM) == []
    # Photon got the trimmed text.
    assert net.calls(PHOTON)[0].url.params["q"] == "houston"


def test_non_texas_results_are_dropped(client, geo):
    net, headers = geo
    r = client.get("/api/geocode", params={"q": "shreveport", "mode": "suggest"}, headers=headers)
    assert r.status_code == 200 and r.json() == []


def test_submit_falls_back_to_nominatim(client, geo):
    net, headers = geo
    r = client.get("/api/geocode", params={"q": "old town spring", "mode": "submit"},
                   headers=headers)
    assert r.status_code == 200
    assert r.json()[0]["display_name"].startswith("Old Town Spring")
    assert len(net.calls(NOMINATIM)) == 1
    params = net.calls(NOMINATIM)[0].url.params
    assert params["bounded"] == "1" and params["countrycodes"] == "us"


def test_submit_skips_nominatim_when_photon_finds_it(client, geo):
    net, headers = geo
    r = client.get("/api/geocode", params={"q": "houston", "mode": "submit"}, headers=headers)
    assert r.status_code == 200 and r.json()[0]["name"] == "Houston"
    assert net.calls(NOMINATIM) == []


def test_nominatim_is_throttled(client, geo):
    net, headers = geo
    first = client.get("/api/geocode", params={"q": "nowhere one", "mode": "submit"},
                       headers=headers)
    assert first.status_code == 200
    # Within 1.1 s of the last call: skipped, not sent.
    second = client.get("/api/geocode", params={"q": "nowhere two", "mode": "submit"},
                        headers=headers)
    assert len(net.calls(NOMINATIM)) == 1
    assert second.status_code == 502
    # A skipped fallback is not cached as "no such place".
    geocode._nominatim_state["last"] = 0.0
    third = client.get("/api/geocode", params={"q": "nowhere two", "mode": "submit"},
                       headers=headers)
    assert third.status_code == 200 and third.json()
    assert len(net.calls(NOMINATIM)) == 2


def test_query_length_and_mode_validation(client, geo):
    net, headers = geo
    assert client.get("/api/geocode", params={"q": "x" * 101}, headers=headers).status_code == 422
    # The limit applies after trimming.
    ok = client.get("/api/geocode", params={"q": "  " + "x" * 100 + "  "}, headers=headers)
    assert ok.status_code == 200
    assert client.get("/api/geocode", params={"q": "houston", "mode": "bogus"},
                      headers=headers).status_code == 422
    assert client.get("/api/geocode", headers=headers).status_code == 422


def test_per_ip_rate_limit(client, geo):
    net, headers = geo
    assert (limits.geocode_limiter.limit, limits.geocode_limiter.window_s) == (60, 60.0)
    codes = [client.get("/api/geocode", params={"q": "houston"}, headers=headers).status_code
             for _ in range(60)]
    assert codes == [200] * 60
    limited = client.get("/api/geocode", params={"q": "houston"}, headers=headers)
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1
    other = client.get("/api/geocode", params={"q": "houston"},
                       headers={"X-Forwarded-For": next(_ips)})
    assert other.status_code == 200
    # Repeats were served from cache: one upstream call for 60 requests.
    assert len(net.calls(PHOTON)) == 1


def test_upstream_outage_is_5xx_not_empty(client, geo):
    net, headers = geo
    net.routes.pop(PHOTON)
    r = client.get("/api/geocode", params={"q": "austin"}, headers=headers)
    assert r.status_code == 502
