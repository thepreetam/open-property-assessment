"""
Market intelligence: uplift/cost by zip. Uses real Zillow data when ZILLOW_API_KEY set, else stub.
"""
from typing import Any, Dict, Optional, Tuple

from core.config import settings

# Stub: zip prefix → cost multiplier, uplift multiplier (1.0 = use base repair_data)
_ZIP_MULTIPLIERS: Dict[str, Tuple[float, float]] = {
    "85": (1.15, 1.08),   # AZ example
    "90": (1.20, 1.05),
    "10": (1.25, 1.10),   # MA/NY metro
    "75": (0.95, 0.98),   # TX example
}


def get_market_multipliers(zip_code: Optional[str]) -> Tuple[float, float]:
    """Return (cost_multiplier, uplift_multiplier) for zip. Uses real data when ZILLOW_API_KEY set."""
    if not zip_code or len(zip_code) < 2:
        return (1.0, 1.0)
    if settings.zillow_api_key:
        try:
            from core.real_data import get_zillow_data
            data = get_zillow_data(zip_code.strip())
            if data:
                return (float(data.get("cost_multiplier", 1.0)), float(data.get("uplift_multiplier", 1.0)))
        except Exception:
            pass
    prefix = zip_code[:2]
    return _ZIP_MULTIPLIERS.get(prefix, (1.0, 1.0))


def get_repair_uplift_stub(
    repair_type: str,
    zip_code: Optional[str],
    base_home_value: int,
    base_uplift_pct: float,
) -> Dict[str, Any]:
    """Stub: estimated uplift for repair in location. Real impl would use comps."""
    cost_mult, uplift_mult = get_market_multipliers(zip_code)
    adjusted_uplift_pct = base_uplift_pct * uplift_mult
    estimated_uplift = int(base_home_value * adjusted_uplift_pct)
    return {
        "estimated_uplift": estimated_uplift,
        "uplift_pct": round(adjusted_uplift_pct, 4),
        "confidence_interval": "stub",
        "sample_size": 0,
        "recent_trend": None,
    }
