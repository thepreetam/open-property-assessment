"""
Webhooks (Phase 4): notify external URLs on job completion.
"""
import asyncio
from typing import List, Optional

try:
    import aiohttp
except ImportError:
    aiohttp = None


async def notify_webhook(url: str, payload: dict) -> bool:
    """POST payload to URL. Returns True if 2xx."""
    if not aiohttp:
        return False
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=5) as resp:
                return 200 <= resp.status < 300
    except Exception:
        return False


async def notify_webhooks(urls: List[str], payload: dict) -> List[bool]:
    """Notify all URLs; return list of success."""
    if not urls:
        return []
    results = await asyncio.gather(*[notify_webhook(u, payload) for u in urls])
    return list(results)
