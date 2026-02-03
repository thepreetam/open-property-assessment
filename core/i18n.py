"""
i18n stub (Phase 4): locale param and string lookup. Region-specific cost/uplift in market_intelligence.
"""
from typing import Dict, Optional

# Stub: key -> { locale -> text }
_STRINGS: Dict[str, Dict[str, str]] = {
    "strategy.budget_flip": {"en": "Quick flip", "es": "Venta rápida"},
    "strategy.value_add": {"en": "Value add", "es": "Añadir valor"},
    "strategy.premium_reno": {"en": "Premium reno", "es": "Reforma premium"},
    "recommendation.reason": {"en": "Highest ROI", "es": "Mayor ROI"},
}


def t(key: str, locale: str = "en") -> str:
    """Return translated string for key; fallback to en."""
    if key not in _STRINGS:
        return key
    return _STRINGS[key].get(locale, _STRINGS[key].get("en", key))


def get_locale(accept_language: Optional[str] = None) -> str:
    """Parse Accept-Language header; return first supported locale (en, es)."""
    if not accept_language:
        return "en"
    for part in accept_language.split(","):
        part = part.strip().split(";")[0].lower()
        if part.startswith("es"):
            return "es"
        if part.startswith("en"):
            return "en"
    return "en"
