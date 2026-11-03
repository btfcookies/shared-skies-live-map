"""Shared fixtures for the backend test suite.

The suite runs offline and without a PurpleAir key, the same way CI does.
Every httpx.AsyncClient the backend creates gets a mock transport routed
through OFFLINE_NET: a test registers a handler per host, and any request
with no handler fails as a connection error (what a real outage looks like
to the backend). Nothing can reach PurpleAir, NOAA, Open-Meteo or the
geocoders, so no test spends PurpleAir points.
"""
import inspect
import os
import shutil
import tempfile

# Environment first: backend.main reads it at import time.
_SNAPSHOTS_DIR = tempfile.mkdtemp(prefix="ssi-test-snapshots-")
os.environ["SSI_SNAPSHOTS_DIR"] = _SNAPSHOTS_DIR
for _var in (
    "PURPLEAIR_API_KEY", "KEEPALIVE_URL", "RENDER_EXTERNAL_URL",
    "UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN", "ALLOWED_ORIGINS",
    # Code defaults only, so the tests check the shipped cadence.
    "PURPLEAIR_CACHE_TTL_MIN", "PURPLEAIR_MAX_AGE_MIN", "QUANTUM_CACHE_TTL_MIN",
    "OPENMETEO_GRID_TTL_MIN", "CONDITIONS_TTL_MIN", "CLICK_WEATHER_TTL_MIN",
):
    os.environ.pop(_var, None)

import httpx  # noqa: E402
import pytest  # noqa: E402


class OfflineNet:
    """Per-host request router for the mock transport."""

    def __init__(self):
        self.routes = {}
        self.requests = []

    def route(self, host, handler):
        """handler(request) -> httpx.Response (sync or async)."""
        self.routes[host] = handler

    def calls(self, host):
        return [r for r in self.requests if r.url.host == host]

    def reset(self):
        self.routes.clear()
        self.requests.clear()

    async def handle(self, request):
        self.requests.append(request)
        handler = self.routes.get(request.url.host)
        if handler is None:
            raise httpx.ConnectError(
                f"offline test suite: no route for {request.url.host}", request=request)
        resp = handler(request)
        if inspect.isawaitable(resp):
            resp = await resp
        return resp


OFFLINE_NET = OfflineNet()


@pytest.fixture(scope="session", autouse=True)
def _offline_httpx():
    real_init = httpx.AsyncClient.__init__

    def init(self, *args, **kwargs):
        # Clients built with an explicit transport (a test's own ASGI client)
        # keep it; everything the backend builds goes through OFFLINE_NET.
        if kwargs.get("transport") is None:
            kwargs["transport"] = httpx.MockTransport(OFFLINE_NET.handle)
            kwargs["trust_env"] = False
        real_init(self, *args, **kwargs)

    mp = pytest.MonkeyPatch()
    mp.setattr(httpx.AsyncClient, "__init__", init)
    yield
    mp.undo()
    shutil.rmtree(_SNAPSHOTS_DIR, ignore_errors=True)


@pytest.fixture
def net():
    """The offline router, cleared before and after the test."""
    OFFLINE_NET.reset()
    yield OFFLINE_NET
    OFFLINE_NET.reset()


@pytest.fixture
def no_snapshots(monkeypatch):
    """Keep a test's fake data out of the snapshot dir the app hydrates from."""
    from backend import snapshots

    async def _skip(path, data):
        return None

    monkeypatch.setattr(snapshots, "save_async", _skip)


@pytest.fixture(scope="session")
def bundle():
    """The deployed model bundle, loaded once for the whole session."""
    from backend import model
    b = model.load_bundle()
    assert b is not None, "models/ensemble.joblib failed to load"
    return b


@pytest.fixture
def with_bundle(monkeypatch, bundle):
    """The bundle installed in main.state, for calling predict functions
    directly without the app."""
    import backend.main as main
    monkeypatch.setitem(main.state, "bundle", bundle)
    return bundle


@pytest.fixture(scope="session")
def tract_lookup():
    """The tract lookup enriched exactly as the lifespan does it. Read-only:
    copy before mutating."""
    import pandas as pd
    from backend import config, model
    df = pd.read_parquet(config.LOOKUP_PATH)
    df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)
    model.enrich_tract_lookup_with_distances(df)
    return df


@pytest.fixture(scope="session")
def client(_offline_httpx, bundle, tmp_path_factory):
    """A TestClient running the real lifespan (model load, tract lookup,
    refresh loop) with no key and no network, after the startup compute and
    the first placement solve have finished."""
    from fastapi.testclient import TestClient
    import backend.main as main
    from backend import model, visits

    mp = pytest.MonkeyPatch()
    mp.setattr(model, "load_bundle", lambda: bundle)
    mp.setattr(visits, "VISITS_DB", str(tmp_path_factory.mktemp("visits") / "visits.sqlite"))
    try:
        with TestClient(main.app) as c:
            # Blocks until the refresh loop's startup compute lands.
            r = c.get("/api/texas/predictions")
            assert r.status_code == 200, r.text
            # The loop solves placement right after; join that solve here so
            # tests never race it.
            r = c.get("/api/quantum/sensor-placement")
            assert r.status_code == 200, r.text
            yield c
    finally:
        mp.undo()


@pytest.fixture(scope="session")
def predictions(client):
    """The startup /api/texas/predictions body."""
    return client.get("/api/texas/predictions").json()
