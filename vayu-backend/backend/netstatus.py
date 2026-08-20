"""Cached network reachability probes — a circuit-breaker for the voice pipeline.

When the network is down (or APIs are blocked), calling Groq/Sarvam would
stall each query for seconds of timeouts. We probe reachability once per TTL
and let the orchestrator skip straight to its fallback path while unreachable.
On a normal network the probe succeeds and the pipeline runs LIVE automatically.

Probes use a real HTTPS GET (not just TCP connect) because some networks
allow the TCP handshake but block TLS/HTTP — exactly what we need to know.
"""
import asyncio
import time

CACHE_TTL = 30.0
_cache: dict = {"ts": 0.0, "groq": None, "sarvam": None}
_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


async def _http_probe(url: str, timeout: float = 3.0, headers: dict = None) -> bool:
    """True if the host answers over HTTPS (any HTTP status counts)."""
    try:
        import httpx
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, verify=True
        ) as client:
            resp = await client.get(url, headers=headers)
            return resp.status_code < 600  # any HTTP response = reachable
    except Exception:
        return False


async def _probe_cached(key: str, url: str, headers: dict = None) -> bool:
    async with _get_lock():
        now = time.time()
        if _cache[key] is None or now - _cache["ts"] > CACHE_TTL:
            _cache[key] = await _http_probe(url, headers=headers)
            _cache["ts"] = time.time()
        return bool(_cache[key])


async def groq_available() -> bool:
    return await _probe_cached("groq", "https://api.groq.com/openai/v1/models")


async def sarvam_available() -> bool:
    import os
    headers = {"api-subscription-key": os.getenv("SARVAM_API_KEY", "")}
    return await _probe_cached("sarvam", "https://api.sarvam.ai/text-to-speech", headers=headers)
