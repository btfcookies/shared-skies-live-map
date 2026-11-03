"""Typo-tolerant place search proxy: Photon first, throttled Nominatim fallback."""

import asyncio
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Request

from backend import limits
from backend.http_client import shared_http
from backend.ratelimit import client_ip

router = APIRouter()


# ── Geocoding (typo-tolerant autocomplete) ───────────────────────────────────
# Primary: Photon (photon.komoot.io) — fuzzy/typo-tolerant, built for
# search-as-you-type ("hueston" still finds Houston). Results are biased to the
# Texas centroid and HARD-filtered to a Texas bounding box.
# Fallback: Nominatim (exact-match), only for an explicit search (mode=submit)
# that Photon found nothing for — scoped to Texas via viewbox+bounded instead
# of appending ", Texas" to the query text. Nominatim's usage policy allows at
# most 1 request/second and forbids autocomplete use, so it is never called
# while typing and is globally throttled.
# Both are normalized to the same shape the frontend expects:
#   [{name, display_name, lat, lon, type}]
_TX_BBOX = (-107.0, 25.5, -93.2, 36.8)  # minLon, minLat, maxLon, maxLat (+margin)
_GEOCODE_UA = "SharedSkiesInitiative/1.0 (+https://sharedskiesinitiative.org; chebrolusaketh@gmail.com)"
_geocode_cache: OrderedDict = OrderedDict()   # (q, lang, mode) -> {results, expires}
_GEOCODE_CACHE_TTL_MIN = 24 * 60        # places don't move
_GEOCODE_EMPTY_CACHE_TTL_MIN = 15       # a miss may be a typo the user fixes
_GEOCODE_CACHE_MAX = 500
_GEOCODE_MAX_LEN = 100
_NOMINATIM_MIN_INTERVAL_S = 1.1
_nominatim_state = {"last": 0.0, "lock": None}



# "Did you mean" spell-correction layer: Photon's fuzziness handles swapped /
# extra letters but misses dropped letters mid-word ("corpus cristi"). We
# correct the query against the big Texas place names with stdlib difflib and
# re-query. Names only — coordinates always come from the geocoder.
_TX_CITIES = [
    "Houston", "San Antonio", "Dallas", "Austin", "Fort Worth", "El Paso",
    "Arlington", "Corpus Christi", "Plano", "Laredo", "Lubbock", "Garland",
    "Irving", "Amarillo", "Grand Prairie", "Brownsville", "Pasadena",
    "McKinney", "Mesquite", "Killeen", "Frisco", "McAllen", "Waco",
    "Carrollton", "Denton", "Midland", "Abilene", "Beaumont", "Round Rock",
    "Odessa", "Wichita Falls", "Richardson", "Lewisville", "Tyler",
    "College Station", "Pearland", "San Angelo", "Allen", "League City",
    "Sugar Land", "Longview", "Edinburg", "Mission", "Bryan", "Baytown",
    "Pharr", "Temple", "Missouri City", "Flower Mound", "Harlingen",
    "Victoria", "Conroe", "New Braunfels", "Mansfield", "Cedar Park",
    "Rowlett", "Port Arthur", "Euless", "Georgetown", "Pflugerville",
    "DeSoto", "San Marcos", "Grapevine", "Bedford", "Galveston",
    "Cedar Hill", "Texas City", "Wylie", "Keller", "Coppell", "Rockwall",
    "Huntsville", "Duncanville", "Sherman", "The Woodlands", "Texarkana",
    "Friendswood", "Weslaco", "Lufkin", "Nacogdoches", "Del Rio", "Eagle Pass",
    "Katy", "Spring", "Humble", "Cypress", "Kingwood", "Tomball", "Sealy",
    "Boerne", "Kerrville", "Seguin", "Schertz", "Alamo Heights", "Uvalde",
    "Stephenville", "Granbury", "Weatherford", "Cleburne", "Waxahachie",
    "Corsicana", "Ennis", "Athens", "Palestine", "Marshall", "Paris",
    "Greenville", "Denison", "Gainesville", "Mineral Wells", "Big Spring",
    "Snyder", "Sweetwater", "Brownwood", "Kingsville", "Alice", "Beeville",
    "Port Lavaca", "Bay City", "Angleton", "Lake Jackson", "Freeport",
    "Orange", "Vidor", "Silsbee", "Jasper", "Livingston", "Bastrop",
    "Lockhart", "Taylor", "Brenham", "Navasota", "Hearne", "Cameron",
]


def _spell_correct_tx(query: str) -> str | None:
    """Return a corrected Texas place name if the query looks like a typo of
    one, else None. Trailing 'tx'/'texas' tokens are ignored for matching."""
    import difflib
    qc = query.lower().strip()
    for suf in (" texas", " tx", ",texas", ",tx"):
        if qc.endswith(suf):
            qc = qc[: -len(suf)].rstrip(", ")
    if len(qc) < 4:
        return None
    match = difflib.get_close_matches(qc, [c.lower() for c in _TX_CITIES], n=1, cutoff=0.78)
    if not match:
        return None
    corrected = next(c for c in _TX_CITIES if c.lower() == match[0])
    return corrected if corrected.lower() != qc else None


def _photon_to_result(feat: dict) -> dict | None:
    """Normalize one Photon GeoJSON feature to the frontend's result shape."""
    try:
        p = feat.get("properties", {})
        # The TX bbox has margin that clips border-state slivers — hard-reject
        # anything Photon labels with a non-Texas state.
        st = p.get("state")
        if st and st not in ("Texas", "TX"):
            return None
        lon, lat = feat["geometry"]["coordinates"][:2]
        name = p.get("name") or " ".join(
            x for x in (p.get("housenumber"), p.get("street")) if x
        ) or p.get("city") or p.get("postcode")
        if not name:
            return None
        parts = []
        street = " ".join(x for x in (p.get("housenumber"), p.get("street")) if x)
        if street and street != name:
            parts.append(street)
        for k in ("district", "city", "county", "state"):
            v = p.get(k)
            if v and v != name and v not in parts:
                parts.append(v)
        if p.get("postcode"):
            parts.append(p["postcode"])
        display = name + (", " + ", ".join(parts) if parts else "")
        return {
            "name": str(name),
            "display_name": display,
            "lat": str(lat),
            "lon": str(lon),
            "type": p.get("osm_value") or p.get("type") or "place",
        }
    except Exception:
        return None


def _geocode_lang(accept_language: str) -> str:
    """Primary language subtag of Accept-Language ('es-MX,es;q=0.9' -> 'es'),
    so the cache key stays small and bounded."""
    tag = (accept_language or "").split(",")[0].split(";")[0].strip().lower()
    tag = tag.split("-")[0]
    return tag if tag.isalpha() and len(tag) <= 3 else "en"


async def _nominatim_search(query: str, lang: str) -> list | None:
    """Nominatim fallback, globally throttled to one call per 1.1 s. Returns
    None when skipped (throttled, or another call in flight) or on error."""
    if _nominatim_state["lock"] is None:
        _nominatim_state["lock"] = asyncio.Lock()
    lock = _nominatim_state["lock"]
    if lock.locked() or time.monotonic() - _nominatim_state["last"] < _NOMINATIM_MIN_INTERVAL_S:
        return None
    minlon, minlat, maxlon, maxlat = _TX_BBOX
    async with lock:
        _nominatim_state["last"] = time.monotonic()
        try:
            r = await shared_http().get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": query, "format": "json", "countrycodes": "us",
                    "limit": 6, "viewbox": f"{minlon},{maxlat},{maxlon},{minlat}",
                    "bounded": 1,
                },
                headers={"User-Agent": _GEOCODE_UA, "Accept-Language": lang},
                timeout=8.0,
            )
            r.raise_for_status()
            return [{
                "name": (it.get("display_name") or "").split(",")[0],
                "display_name": it.get("display_name", ""),
                "lat": it.get("lat"), "lon": it.get("lon"),
                "type": it.get("type", "place"),
            } for it in r.json()]
        except Exception as e:
            print(f"[geocode] nominatim failed: {type(e).__name__}: {e}")
            return None


@router.get("/api/geocode")
async def proxy_geocode(
    request: Request,
    q: str,
    mode: Literal["suggest", "submit"] = "suggest",
    accept_language: str = Header(default="en"),
):
    """Texas place search. mode=suggest (while typing) uses Photon only;
    mode=submit (Enter / explicit search) adds the Nominatim fallback when
    Photon finds nothing. 60 requests/min per client IP."""
    allowed, retry_after = limits.geocode_limiter.check(client_ip(request))
    if not allowed:
        raise HTTPException(429, "Too many requests.", headers={"Retry-After": str(retry_after)})

    query = q.strip()
    if len(query) > _GEOCODE_MAX_LEN:
        raise HTTPException(422, f"q must be at most {_GEOCODE_MAX_LEN} characters.")
    # The old frontend appended ", Texas" to every query; strip it so fuzzy
    # matching sees the user's actual words (Texas scoping is done by bbox).
    if query.lower().endswith(", texas"):
        query = query[:-7].rstrip(", ")
    if not query:
        return []

    # Serve recent identical queries from cache (autocomplete hammers this).
    lang = _geocode_lang(accept_language)
    ck = (query.lower(), lang, mode)
    now = datetime.now(timezone.utc)
    hit = _geocode_cache.get(ck)
    if hit and now < hit["expires"]:
        _geocode_cache.move_to_end(ck)
        return hit["results"]

    results: list = []
    minlon, minlat, maxlon, maxlat = _TX_BBOX
    upstream_failed = False
    throttled = 0  # Retry-After seconds once the global Photon budget is spent

    async def _photon_search(q_text: str) -> list:
        nonlocal upstream_failed, throttled
        allowed, wait = limits.photon_budget.check(limits.GLOBAL_KEY)
        if not allowed:
            throttled = wait
            return []
        try:
            r = await shared_http().get(
                "https://photon.komoot.io/api",
                params={
                    "q": q_text,
                    "limit": 10,
                    "lat": 31.0, "lon": -99.3,      # bias: Texas centroid
                    "bbox": f"{minlon},{minlat},{maxlon},{maxlat}",
                },
                headers={"User-Agent": _GEOCODE_UA},
                timeout=6.0,
            )
            r.raise_for_status()
            return [res for f in r.json().get("features", [])
                    if (res := _photon_to_result(f))]
        except Exception as e:
            upstream_failed = True
            print(f"[geocode] photon failed: {type(e).__name__}: {e}")
            return []

    def _merge(*lists) -> list:
        """Concatenate, dedupe near-identical entries, cap at 6."""
        out, seen = [], set()
        for lst in lists:
            for res in lst:
                key = (res["name"].lower(),
                       round(float(res["lat"]), 3), round(float(res["lon"]), 3))
                if key in seen:
                    continue
                seen.add(key)
                out.append(res)
                if len(out) >= 6:
                    return out
        return out

    # 1) Photon as typed — typo-tolerant, autocomplete-grade.
    raw = await _photon_search(query)

    # 2) "Did you mean": if the query looks like a misspelled Texas place,
    #    re-query with the corrected name. Empty as-typed results -> the
    #    correction leads; otherwise the as-typed top hit stays first and
    #    the corrected city is injected right behind it (Google-style mix).
    corrected_name = _spell_correct_tx(query)
    if corrected_name:
        fixed = await _photon_search(corrected_name)
        if not raw:
            results = _merge(fixed, raw)
        else:
            results = _merge(raw[:1], fixed[:2], raw[1:])
    else:
        results = _merge(raw)

    # 3) Nominatim fallback (exact match), explicit searches only.
    if not results and mode == "submit":
        nominatim = await _nominatim_search(query, lang)
        if nominatim is None:
            upstream_failed = True
        else:
            results = nominatim

    if not results and throttled:
        # Global Photon budget spent: not an answer, so not cached either.
        raise HTTPException(429, "Too many requests.", headers={"Retry-After": str(throttled)})
    if not results and upstream_failed:
        # Don't cache an outage as "no such place".
        raise HTTPException(status_code=502, detail="Geocoding service unavailable.")

    # Partial answers (one Photon call failed) get the short TTL too.
    ttl = (_GEOCODE_CACHE_TTL_MIN if results and not (upstream_failed or throttled)
           else _GEOCODE_EMPTY_CACHE_TTL_MIN)
    _geocode_cache[ck] = {"results": results, "expires": now + timedelta(minutes=ttl)}
    if len(_geocode_cache) > _GEOCODE_CACHE_MAX:
        _geocode_cache.popitem(last=False)
    return results
