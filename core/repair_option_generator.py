"""
RepairOptionGenerator-style: multiple options per defect, feasibility, ROI-sorted.
Extends repair_data with optional market adjustment and filters.
"""
from typing import Any, Dict, List, Optional

from repair_data import REPAIR_OPTIONS_BY_CATEGORY
from core.market_intelligence import get_market_multipliers
from core.repair_engine import calculate_roi


def _feasible(option: dict, context: Optional[Dict[str, Any]] = None) -> bool:
    """Feasibility filter. Context can include max_budget, timeline_days, skill_available."""
    if not context:
        return True
    max_budget = context.get("max_budget")
    if max_budget is not None:
        lo, hi = option["cost_range"]
        if (lo + hi) // 2 > max_budget:
            return False
    return True


def generate_options_for_finding(
    defect_category: str,
    base_home_value: int,
    zip_code: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    All repair options for a finding category, filtered by feasibility, ROI-sorted.
    Optionally applies market cost/uplift multipliers by zip.
    """
    options = REPAIR_OPTIONS_BY_CATEGORY.get(defect_category, [])
    cost_mult, uplift_mult = get_market_multipliers(zip_code)
    out = []
    for opt in options:
        if not _feasible(opt, context):
            continue
        lo, hi = opt["cost_range"]
        adjusted_opt = {
            **opt,
            "cost_range": (int(lo * cost_mult), int(hi * cost_mult)),
            "uplift_pct": opt["uplift_pct"] * uplift_mult,
        }
        roi = calculate_roi(adjusted_opt, base_home_value)
        out.append({
            "category": defect_category,
            "option": opt["label"],
            "option_name": opt["name"],
            "strategy_tier": opt["strategy_tier"],
            "time_days": opt["time_days"],
            "skill_required": opt["skill_required"],
            **roi,
        })
    out.sort(key=lambda x: x["roi_pct"], reverse=True)
    return out


def build_timeline_tasks(strategy_repairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simple timeline: one task per repair with start_day and duration_days (sequential)."""
    tasks = []
    start = 0
    for r in strategy_repairs:
        duration = int(r.get("time_days", 1))
        tasks.append({
            "category": r.get("category", ""),
            "option": r.get("option", ""),
            "start_day": start,
            "duration_days": duration,
            "estimated_cost": r.get("estimated_cost", 0),
        })
        start += duration
    return tasks
