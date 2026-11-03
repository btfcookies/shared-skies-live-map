"""Health check and city list routes."""

from datetime import datetime, timezone

from fastapi import APIRouter

from backend import purpleair_feed, weather_feed
from backend.config import CITIES, PURPLEAIR_CACHE_TTL_MIN
from backend.purpleair_feed import purpleair_key_configured
from backend.state import state
from backend.util import age_min, rss_mb


router = APIRouter()


@router.get("/api/health")
async def health():
    """Health + live-data-source status. The data_sources block makes silent
    degradation visible: if PurpleAir/HMS/weather quietly fail, predictions
    fall back to climatology/medians — operators need to SEE that. Reads
    in-memory state only, so it is safe as Render's health check. HTTP status
    is always 200: a degraded map is still worth serving, and a non-200 would
    make Render restart the instance."""
    now = datetime.now(timezone.utc)
    pa = purpleair_feed.purpleair_cache.get("data") or {}
    texas = state.get("cache_texas", {}).get("data") or {}
    bundle = state.get("bundle")
    sources = texas.get("data_sources", {})
    live_neighbors = bool(sources.get("using_live_neighbors"))
    predictions_age = age_min(texas.get("generated_at"), now)
    purpleair_age = age_min(pa.get("fetched_at"), now) if pa.get("usable") else None
    degraded = (
        bundle is None
        or not live_neighbors
        or predictions_age is None
        or predictions_age > 3 * PURPLEAIR_CACHE_TTL_MIN
    )
    return {
        "status": "degraded" if degraded else "ok",
        "model_loaded": bundle is not None,
        "model_version": (bundle or {}).get("version"),
        # The version string is unchanged across retrains, so it cannot tell you
        # WHICH bundle is live. The feature count can: the EJScreen removal took
        # the model from 38 to 30, so this is the field to check after a deploy.
        "model_features": len((bundle or {}).get("feature_names", [])),
        "lookup_loaded": state.get("tract_lookup") is not None,
        "available_cities": list(CITIES.keys()),
        "live_purpleair_sensors": len(pa.get("sensors", [])) if pa.get("usable") else 0,
        "purpleair_fetched_at": pa.get("fetched_at"),
        "purpleair_key_configured": purpleair_key_configured(),
        "live_neighbors": live_neighbors,
        "predictions_generated_at": texas.get("generated_at"),
        "predictions_age_min": round(predictions_age, 1) if predictions_age is not None else None,
        "purpleair_age_min": round(purpleair_age, 1) if purpleair_age is not None else None,
        "refresh_interval_min": PURPLEAIR_CACHE_TTL_MIN,
        "memory_rss_mb": rss_mb(),
        "conditions_fetched_at": (weather_feed.conditions_cache.get("data") or {}).get("fetched_at"),
        "data_sources": sources,
    }


@router.get("/api/cities")
async def list_cities():
    """List available cities with metadata."""
    return {
        "cities": [
            {
                "id": city_id,
                "name": config["display_name"],
                "center": config["center"],
                "fips": config["fips"]
            }
            for city_id, config in CITIES.items()
        ]
    }
