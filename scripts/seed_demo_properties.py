"""
Seed 5 demo properties with realistic Job results for acquisition demos.
Run: DATABASE_URL=postgresql://... python scripts/seed_demo_properties.py
Creates one team, one workspace, 5 properties, and one completed job per property with result payload.
"""
import os
import sys
import uuid

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.models import Team, Workspace, Property, Job, get_session_factory
from core.config import settings

DEMO_PROPERTIES = [
    {"address": "123 Oak St, Phoenix, AZ 85001", "zip_code": "85001", "home_value": 385000},
    {"address": "456 Elm Ave, Scottsdale, AZ 85251", "zip_code": "85251", "home_value": 620000},
    {"address": "789 Pine Rd, Austin, TX 78701", "zip_code": "78701", "home_value": 545000},
    {"address": "101 Maple Dr, Boston, MA 02101", "zip_code": "02101", "home_value": 890000},
    {"address": "202 Cedar Ln, Dallas, TX 75201", "zip_code": "75201", "home_value": 410000},
]

# Realistic result payload (strategies + recommendation) for demo
def _demo_result(home_value: int, zip_code: str) -> dict:
    return {
        "all_data": [
            {"Photo": 1, "Room": "kitchen", "Quality excerpt": "Outdated cabinets, laminate counters.", "Adjustment $": -18000, "% Adj": -4.0, "Notes": "Outdated kitchen; Repair needs detected"},
            {"Photo": 2, "Room": "bathroom", "Quality excerpt": "Dated fixtures, older tile.", "Adjustment $": -13500, "% Adj": -3.0, "Notes": "Bathroom not notably modern"},
        ],
        "strategies": {
            "budget_flip": {"name": "Quick flip", "philosophy": "Minimal cost for fast sale", "repairs": [], "cost": 1200, "uplift": 25000, "timeline_days": 4, "roi_pct": 1983.3},
            "value_add": {"name": "Value add", "philosophy": "Best ROI per dollar", "repairs": [], "cost": 4500, "uplift": 58000, "timeline_days": 8, "roi_pct": 1187.8},
            "premium_reno": {"name": "Premium reno", "philosophy": "Full modernization", "repairs": [], "cost": 18500, "uplift": 95000, "timeline_days": 22, "roi_pct": 413.5},
        },
        "recommendation": {"strategy_name": "Quick flip", "reason": "Highest ROI (1983.3%) at $1,200 cost."},
    }


def main():
    if not settings.database_url:
        print("Set DATABASE_URL to run seed.")
        sys.exit(1)
    factory = get_session_factory(settings.database_url)
    session = factory()
    try:
        team_id = str(uuid.uuid4())
        ws_id = str(uuid.uuid4())
        session.add(Team(id=team_id, name="Demo Team"))
        session.add(Workspace(id=ws_id, team_id=team_id, name="Demo Workspace"))
        for row in DEMO_PROPERTIES:
            prop_id = str(uuid.uuid4())
            session.add(Property(
                id=prop_id,
                workspace_id=ws_id,
                address=row["address"],
                zip_code=row["zip_code"],
                home_value=str(row["home_value"]),
            ))
            job_id = str(uuid.uuid4())
            session.add(Job(
                id=job_id,
                property_id=prop_id,
                workspace_id=ws_id,
                status="completed",
                result=_demo_result(row["home_value"], row["zip_code"]),
            ))
        session.commit()
        print(f"Created team {team_id}, workspace {ws_id}, 5 properties, 5 completed jobs.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
