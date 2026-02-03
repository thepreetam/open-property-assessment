"""
Real data integration: Zillow (or RapidAPI Zillow) for comps/uplift by zip.
When ZILLOW_API_KEY (or RAPIDAPI_KEY) is set, market_intelligence uses this; otherwise stub.
Results cached by zip to respect rate limits (24h TTL).
"""
from typing import Any, Dict, Optional

from core.config import settings
from core.cache import get_cached_result, set_cached_result

# National median reference (approximate; used to derive zip multiplier when API returns median)
_NATIONAL_MEDIAN_REF = 420000.0
_CACHE_PREFIX = "market"
_CACHE_TTL = 86400  # 24h


def _fetch_zillow_by_zip(zip_code: str) -> Optional[Dict[str, Any]]:
    """
    Call Zillow/RapidAPI for zip-level data. Returns dict with median_value or similar.
    RapidAPI Zillow endpoints vary; this uses a generic pattern: GET with zip, parse median/estimate.
    """
    if not settings.zillow_api_key or len(zip_code) < 5:
        return None
    try:
        import urllib.request
        import json
        # RapidAPI Zillow: search by zip returns list; we aggregate prices for multiplier
        url = f"https://zillow-com1.p.rapidapi.com/search?location={zip_code}"
        req = urllib.request.Request(
            url,
            headers={
                "X-RapidAPI-Key": settings.zillow_api_key,
                "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        # Parse: could be list of listings with price, or single property
        return _parse_zillow_response(data, zip_code)
    except Exception:
        return None


def _parse_zillow_response(data: Any, zip_code: str) -> Optional[Dict[str, Any]]:
    """Extract median or average price from Zillow response; shape varies by endpoint."""
    if not data:
        return None
    prices = []
    if isinstance(data, list):
        for item in data:
            p = item.get("price") or item.get("zestimate") or item.get("unformattedPrice")
            if p is not None:
                try:
                    prices.append(int(float(p)))
                except (TypeError, ValueError):
                    pass
    elif isinstance(data, dict):
        p = data.get("price") or data.get("zestimate") or data.get("unformattedPrice")
        if p is not None:
            try:
                prices.append(int(float(p)))
            except (TypeError, ValueError):
                pass
        # Nested results
        for key in ("results", "searchResults", "listings", "properties"):
            if key in data and isinstance(data[key], list):
                for item in data[key]:
                    p = item.get("price") or item.get("zestimate") or item.get("unformattedPrice")
                    if p is not None:
                        try:
                            prices.append(int(float(p)))
                        except (TypeError, ValueError):
                            pass
    if not prices:
        return None
    median_val = sorted(prices)[len(prices) // 2] if prices else None
    if median_val is None:
        return None
    return {"median_value": median_val, "sample_size": len(prices), "zip_code": zip_code}


def get_zillow_data(zip_code: str) -> Optional[Dict[str, Any]]:
    """
    Get market data for zip: try cache, then API; return dict with uplift_multiplier, cost_multiplier.
    Returns None on miss/error (caller falls back to stub).
    """
    if not zip_code or len(zip_code) < 5:
        return None
    cache_key = f"{_CACHE_PREFIX}:zip:{zip_code}"
    cached = get_cached_result(cache_key)
    if cached is not None:
        return cached
    raw = _fetch_zillow_by_zip(zip_code)
    if raw is None:
        return None
    # Derive multipliers: uplift_mult = zip_median / national_ref (hot markets > 1)
    median = raw.get("median_value")
    if median is None:
        return None
    uplift_mult = round(median / _NATIONAL_MEDIAN_REF, 4)
    uplift_mult = max(0.5, min(2.0, uplift_mult))  # clamp
    cost_mult = uplift_mult  # assume cost tracks with market
    result = {"uplift_multiplier": uplift_mult, "cost_multiplier": cost_mult, "median_value": median, "sample_size": raw.get("sample_size", 0)}
    set_cached_result(cache_key, result, _CACHE_TTL)
    return result
