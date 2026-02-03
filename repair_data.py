"""
Repair options by finding category for the Repair ROI Optimizer.
Static data: cost ranges, time, strategy tier, uplift % of home value (Phase 1; no external API).
"""

FINDING_CATEGORIES = ["kitchen_cabinets", "bathroom_fixtures", "exterior", "flooring"]

# Per category: list of repair options. uplift_pct = estimated value uplift as fraction of base_home_value.
REPAIR_OPTIONS_BY_CATEGORY = {
    "kitchen_cabinets": [
        {
            "name": "full_replacement",
            "label": "Full cabinet replacement",
            "cost_range": (2800, 4200),
            "time_days": 5,
            "skill_required": "carpenter",
            "strategy_tier": "premium",
            "uplift_pct": 0.022,
        },
        {
            "name": "refinish_paint",
            "label": "Refinish / paint",
            "cost_range": (600, 1200),
            "time_days": 2,
            "skill_required": "handyman",
            "strategy_tier": "value",
            "uplift_pct": 0.012,
        },
        {
            "name": "hardware_only",
            "label": "Hardware only",
            "cost_range": (200, 400),
            "time_days": 0.5,
            "skill_required": "DIY",
            "strategy_tier": "budget",
            "uplift_pct": 0.005,
        },
    ],
    "bathroom_fixtures": [
        {
            "name": "full_remodel",
            "label": "Full bathroom remodel",
            "cost_range": (5000, 8000),
            "time_days": 7,
            "skill_required": "contractor",
            "strategy_tier": "premium",
            "uplift_pct": 0.045,
        },
        {
            "name": "partial_update",
            "label": "Partial update (vanity, tile)",
            "cost_range": (1500, 3000),
            "time_days": 3,
            "skill_required": "handyman",
            "strategy_tier": "value",
            "uplift_pct": 0.025,
        },
        {
            "name": "cosmetics",
            "label": "Cosmetics (paint, caulk, fixtures)",
            "cost_range": (400, 800),
            "time_days": 1,
            "skill_required": "handyman",
            "strategy_tier": "budget",
            "uplift_pct": 0.01,
        },
    ],
    "exterior": [
        {
            "name": "siding_repair",
            "label": "Siding / facade repair",
            "cost_range": (4000, 7000),
            "time_days": 10,
            "skill_required": "contractor",
            "strategy_tier": "premium",
            "uplift_pct": 0.04,
        },
        {
            "name": "paint_trim",
            "label": "Paint and trim",
            "cost_range": (1500, 3000),
            "time_days": 4,
            "skill_required": "painter",
            "strategy_tier": "value",
            "uplift_pct": 0.02,
        },
        {
            "name": "cleanup_landscaping",
            "label": "Cleanup and landscaping",
            "cost_range": (300, 700),
            "time_days": 1,
            "skill_required": "handyman",
            "strategy_tier": "budget",
            "uplift_pct": 0.008,
        },
    ],
    "flooring": [
        {
            "name": "premium_flooring",
            "label": "Premium flooring",
            "cost_range": (4000, 8000),
            "time_days": 5,
            "skill_required": "contractor",
            "strategy_tier": "premium",
            "uplift_pct": 0.035,
        },
        {
            "name": "refinish_replace",
            "label": "Refinish or replace",
            "cost_range": (2000, 4000),
            "time_days": 3,
            "skill_required": "contractor",
            "strategy_tier": "value",
            "uplift_pct": 0.02,
        },
        {
            "name": "deep_clean_patch",
            "label": "Deep clean and patch",
            "cost_range": (300, 800),
            "time_days": 1,
            "skill_required": "handyman",
            "strategy_tier": "budget",
            "uplift_pct": 0.006,
        },
    ],
}

# Map pipeline "notes" / conditions to finding categories (for get_findings)
NOTE_TO_CATEGORY = {
    "outdated kitchen": "kitchen_cabinets",
    "repair needs detected": "kitchen_cabinets",
    "bathroom not notably modern": "bathroom_fixtures",
    "exterior wear/damage": "exterior",
    "worn flooring": "flooring",
    "flooring": "flooring",
}
