"""Rate limiters and request budgets, one instance each, shared by the routers."""

from backend.ratelimit import FixedWindowLimiter, RecentKeys


# One counted visit per client IP per window; the frontend also sends at most
# one POST per browser session.
VISIT_DEDUPE_MIN = 30


# Per-client-IP limits (see backend/ratelimit.py).
geocode_limiter = FixedWindowLimiter(60, 60.0)
tract_limiter = FixedWindowLimiter(120, 60.0)
visit_dedupe = RecentKeys(VISIT_DEDUPE_MIN * 60.0)


# Global budgets that hold whatever X-Forwarded-For says (a direct caller can
# rotate it past every per-IP limit). Each is one shared key.
GLOBAL_KEY = "*"
photon_budget = FixedWindowLimiter(120, 60.0, max_keys=1)        # uncached Photon calls
visit_budget = FixedWindowLimiter(30, 60.0, max_keys=1)          # visit increments


# Uncached click-weather fetches share the api.open-meteo.com daily quota with
# the weather and conditions grids; 200/h caps them at 4,800 of ~10,000/day.
click_weather_budget = FixedWindowLimiter(200, 3600.0, max_keys=1)
