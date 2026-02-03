"""
Repair strategy engine: findings → strategies + recommendation.
Used by both Streamlit and the API/worker pipeline.
"""
from repair_data import NOTE_TO_CATEGORY, REPAIR_OPTIONS_BY_CATEGORY


def get_findings(per_photo_data: list) -> list:
    """Convert per-photo pipeline output (Room, Notes) into deduped structured findings."""
    seen = set()
    findings = []
    for row in per_photo_data:
        room = row.get("Room", "")
        notes_str = row.get("Notes", "")
        for note in notes_str.split(";"):
            note = note.strip().lower()
            if not note:
                continue
            category = NOTE_TO_CATEGORY.get(note)
            if category and category not in seen:
                seen.add(category)
                findings.append({"category": category, "condition": note, "room": room, "source": "description"})
    return findings


def calculate_roi(option: dict, base_home_value: int) -> dict:
    """Single repair option → cost (mid of range), uplift, roi_pct."""
    lo, hi = option["cost_range"]
    cost = (lo + hi) // 2
    uplift = int(base_home_value * option["uplift_pct"])
    roi_pct = round((uplift - cost) / cost * 100, 1) if cost > 0 else 0.0
    return {"estimated_cost": cost, "estimated_uplift": uplift, "roi_pct": roi_pct}


def generate_strategies(findings: list, base_home_value: int, zip_code: str = None) -> dict:
    """Build budget_flip, value_add, premium_reno strategies and pick recommendation (best ROI)."""
    strategies = {
        "budget_flip": {"name": "Quick flip", "philosophy": "Minimal cost for fast sale", "repairs": [], "cost": 0, "uplift": 0, "timeline_days": 0},
        "value_add": {"name": "Value add", "philosophy": "Best ROI per dollar", "repairs": [], "cost": 0, "uplift": 0, "timeline_days": 0},
        "premium_reno": {"name": "Premium reno", "philosophy": "Full modernization", "repairs": [], "cost": 0, "uplift": 0, "timeline_days": 0},
    }

    for f in findings:
        cat = f["category"]
        options = REPAIR_OPTIONS_BY_CATEGORY.get(cat, [])
        if not options:
            continue
        for strat_key, strat in strategies.items():
            tier = "budget" if strat_key == "budget_flip" else "value" if strat_key == "value_add" else "premium"
            opt = next((o for o in options if o["strategy_tier"] == tier), options[0])
            roi = calculate_roi(opt, base_home_value)
            strat["repairs"].append({"category": cat, "option": opt["label"], "time_days": opt["time_days"], **roi})
            strat["cost"] += roi["estimated_cost"]
            strat["uplift"] += roi["estimated_uplift"]
            strat["timeline_days"] += opt["time_days"]

    for strat in strategies.values():
        strat["roi_pct"] = round((strat["uplift"] - strat["cost"]) / strat["cost"] * 100, 1) if strat["cost"] > 0 else 0.0

    best = max(strategies.values(), key=lambda s: s["roi_pct"])
    recommendation = {"strategy_name": best["name"], "reason": f"Highest ROI ({best['roi_pct']}%) at ${best['cost']:,} cost."}

    return {"strategies": strategies, "recommendation": recommendation}
