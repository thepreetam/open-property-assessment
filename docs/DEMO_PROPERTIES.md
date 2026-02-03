# Demo Properties (Acquisition Readiness)

Five demo properties with before/after and expected ROI for walkthroughs and case studies.

## Seed data

Run the seed script (requires DATABASE_URL):

```bash
DATABASE_URL=postgresql://user:pass@host:5432/db python scripts/seed_demo_properties.py
```

This creates one team, one workspace, 5 properties, and one completed analysis job per property with realistic strategy results.

## Demo property list

| Address | Zip | Home value | Expected ROI highlight |
|---------|-----|------------|------------------------|
| 123 Oak St, Phoenix, AZ 85001 | 85001 | $385,000 | Quick flip ~1983% ROI at $1.2k cost |
| 456 Elm Ave, Scottsdale, AZ 85251 | 85251 | $620,000 | Value add best balance |
| 789 Pine Rd, Austin, TX 78701 | 78701 | $545,000 | Premium reno for full modernization |
| 101 Maple Dr, Boston, MA 02101 | 02101 | $890,000 | High-value market multipliers |
| 202 Cedar Ln, Dallas, TX 75201 | 75201 | $410,000 | Budget flip recommended |

## API usage for demos

- **List properties**: `GET /api/v1/properties?workspace_id=<ws_id>` (use workspace id from seed output).
- **Get strategies**: `GET /api/v1/strategies?property_id=<prop_id>` for any of the 5 property ids.
- **Execute**: `POST /api/v1/execute` with `property_id` and `strategy_key` (e.g. `value_add`).

## Before/after narrative

For video or case studies: "Upload photos of kitchen and bathroom → system detects outdated kitchen and dated bathroom → three strategies (Quick flip, Value add, Premium reno) with cost, uplift, and ROI → recommendation: Quick flip for highest ROI at lowest cost."
