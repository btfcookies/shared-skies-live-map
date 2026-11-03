"""One shared httpx client for outbound calls."""

import httpx


# One pooled client for the small per-request upstream calls (Upstash,
# geocoding), so each call doesn't pay a fresh TLS handshake.
_http_client: dict = {"client": None}


def shared_http() -> httpx.AsyncClient:
    if _http_client["client"] is None:
        _http_client["client"] = httpx.AsyncClient(timeout=8.0)
    return _http_client["client"]


async def close_shared_http() -> None:
    client = _http_client["client"]
    if client is not None:
        await client.aclose()
        _http_client["client"] = None
