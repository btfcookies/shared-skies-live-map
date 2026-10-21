"""Per-client-IP limits (backend/ratelimit.py)."""
import time

from starlette.requests import Request

from backend.ratelimit import FixedWindowLimiter, RecentKeys, client_ip


def _request(headers=None, peer=("198.51.100.7", 5000)):
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "headers": raw, "client": peer})


def test_client_ip_prefers_leftmost_forwarded_for():
    assert client_ip(_request({"X-Forwarded-For": "203.0.113.5, 10.0.0.1"})) == "203.0.113.5"
    assert client_ip(_request({"X-Forwarded-For": "  "})) == "198.51.100.7"
    assert client_ip(_request()) == "198.51.100.7"
    assert client_ip(_request(peer=None)) == "unknown"
    assert len(client_ip(_request({"X-Forwarded-For": "a" * 500}))) == 64


def test_fixed_window_limit_and_retry_after():
    lim = FixedWindowLimiter(3, 60.0)
    results = [lim.check("a") for _ in range(4)]
    assert [ok for ok, _ in results] == [True, True, True, False]
    assert 1 <= results[-1][1] <= 60
    assert lim.check("b") == (True, 0)


def test_fixed_window_resets():
    lim = FixedWindowLimiter(1, 0.05)
    assert lim.check("a")[0] and not lim.check("a")[0]
    time.sleep(0.06)
    assert lim.check("a")[0]


def test_fixed_window_memory_is_bounded():
    lim = FixedWindowLimiter(1, 60.0, max_keys=100)
    for i in range(1000):
        lim.check(str(i))
    assert len(lim._windows) <= 100


def test_recent_keys_dedupe_and_expiry():
    rk = RecentKeys(0.05)
    assert rk.first_time("x") and not rk.first_time("x")
    assert rk.first_time("y")
    time.sleep(0.06)
    assert rk.first_time("x")


def test_recent_keys_memory_is_bounded():
    rk = RecentKeys(60.0, max_keys=50)
    for i in range(500):
        rk.first_time(str(i))
    assert len(rk._seen) <= 51
