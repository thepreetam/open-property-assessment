# Video Walkthrough Script – Repair ROI Optimizer

Use this script to record a short demo (5–8 min) showing upload → analysis → strategies and ROI.

## 1. Intro (30 s)

- "This is the Repair ROI Optimizer: upload photos of a property, get room-by-room condition analysis and three repair strategies with cost, uplift, and ROI."
- "Inspired by Opendoor's Repair Co-Pilot and RiskAI; built for acquisition readiness with real data hooks and ML pipeline."

## 2. Streamlit UI (2 min)

- Open the Streamlit app (local or deployed).
- Sidebar: set **Estimated home value** (e.g. $500,000) and **Zip code** (e.g. 85001).
- **Upload 1–5 photos**: kitchen, bathroom, exterior (use demo images from repo or stock).
- Show: "Loading AI models…" then per-photo: Detected room, Condition analysis, YOLO detection, Suggested adjustment.
- Switch to **Aggregate Summary** tab: table, total adjustment, CSV download.
- Switch to **Repair Strategies** tab: Findings, three strategy cards (Quick flip, Value add, Premium reno), Recommendation.

## 3. API (optional, 1 min)

- Open API docs (`/docs`).
- Show `POST /api/v1/jobs`: upload files + home_value + zip_code → job_id, status_url.
- Show `GET /api/v1/jobs/{job_id}`: status, result with strategies.
- Show `GET /api/v1/strategies?property_id=...` for a demo property.

## 4. ROI and case study (1 min)

- "Recommendation is based on highest ROI at lowest cost; we use static repair data today and can plug in Zillow comps when API key is set."
- Point to case study: "Property X: before/after, $ impact" (see docs/case_studies/).

## 5. Outro (30 s)

- "Next: MLflow for model versioning, load testing for 750 assessments/week, and deployment automation. Link in README."

## Hosting the video

- Upload to YouTube or Loom; add the link to README or docs/DEMO_PROPERTIES.md.
