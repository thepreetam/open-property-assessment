# Repair ROI Optimizer

Opendoor-inspired prototype: upload 1–5 home photos (kitchen, bathroom, living room, exterior), get room classification, condition analysis, and **three repair strategies** (Quick flip, Value add, Premium reno) with estimated cost, uplift, and ROI so you can choose a profit-maximizing path.

## Features

- **Room classification** – Detects room type (kitchen, bathroom, bedroom, exterior, etc.) with confidence
- **Object detection** – YOLO bounding boxes on appliances, furniture, fixtures
- **Condition analysis** – BLIP or LLaVA description → keyword-based quality/repair signals
- **Price adjustment** – Heuristic ±$ and % per photo; aggregate summary with CSV export
- **Repair strategies** – From detected findings (e.g. outdated kitchen, exterior wear), get three strategies:
  - **Quick flip** – Minimal cost, fast sale
  - **Value add** – Best ROI per dollar
  - **Premium reno** – Full modernization
- **Zip code** – Optional market context (Phase 1 uses static cost/uplift data; zip can drive future pricing)
- **Phase 1 API** – FastAPI with health/ready and **POST /api/v1/jobs** (photos + home value → async analysis). Celery + Redis for background processing; PostgreSQL for job status and results. Streamlit UI uses the same shared pipeline (core) in-process.
- **Phase 2** – Properties and portfolio (POST/GET /properties), strategies (GET /strategies by job_id or property_id), execute stub (POST /execute), repair option generator with market stub (zip-based cost/uplift), timeline per strategy, API key auth (API_KEY env + X-API-Key header).
- **Phase 3** – Multi-tenant: teams, workspaces, workspace members (RBAC: admin/member/viewer), audit logs; analytics (GET /analytics/metrics, /executive-summary); contractor API (GET /contractor/repairs, POST /contractor/completion); security notes (docs/SECURITY.md).
- **Phase 4** – Redis cache for results, export (GET /export/job/{id}/csv and /summary), webhooks stub, i18n stub; acquisition package in **docs/** (ARCHITECTURE.md, API-SPEC.md, DATABASE-SCHEMA.md, RUNBOOK.md, SECURITY.md, OPENDOOR-INTEGRATION.md).

## Deploy on Render

**Recommended:** Run the app on Render so it’s always available at a public URL.

1. Push this repo to GitHub (or GitLab/Bitbucket).
2. In the [Render Dashboard](https://dashboard.render.com), click **New → Web Service**, connect the repo.
3. Use these settings (or use the repo’s `render.yaml` Blueprint for one-click deploy):
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0` (Render sets `PORT`; app must bind to `0.0.0.0:$PORT`.)
   - **Python:** Render uses `runtime.txt` (e.g. `python-3.11.7`) if present.
4. **Free and Starter (512 MB RAM):** Both have 512 MB, which is **not enough** to run the AI models. The app will stay up and show an upgrade message when you upload photos. Spins down after 15 min idle (cold start ~1 min).
5. **Standard (2 GB) or higher:** Upgrade in Render → Settings → Instance type (e.g. **Standard $25/mo, 2 GB**). Set env var **`FREE_TIER=false`** to enable room classification, BLIP captioning, YOLO detection, and **Repair Strategies**. Optionally **`USE_LLAVA=true`** for LLaVA (use Standard or higher).
6. **Optional – Hugging Face token:** To remove the "unauthenticated requests" warning and get higher rate limits and faster model downloads, add env var **`HF_TOKEN`** (or `HUGGING_FACE_HUB_TOKEN`) in the Render service with a [Hugging Face read token](https://huggingface.co/settings/tokens).

## Opendoor tie-in

This prototype automates condition flagging and turns findings into repair strategies with illustrative cost/uplift/ROI—inspired by Opendoor’s Repair Co-Pilot and RiskAI. **Not** a production valuation or repair-cost tool.

## Limitations

- Adjustments and repair strategies are **rule-based heuristics** and static cost/uplift data (Phase 1). Not financial or contractor advice.
- Bias: lighting/angle and training data affect results; production would need diverse datasets and real cost data.
- Free and Starter (512 MB) can’t run the models; the app shows an upgrade message. Use Standard (2 GB) or higher and set `FREE_TIER=false` for full analysis and Repair Strategies.

## Run locally (optional)

### Streamlit only (no API/DB)

```bash
pip install -r requirements.txt
streamlit run app.py
```

First run loads models (1–2 min on CPU). Set **estimated home value** and optional **zip code** in the sidebar; upload photos to see per-photo details, aggregate summary, and the **Repair Strategies** tab.

### Full stack (API + worker + PostgreSQL + Redis)

From the repo root:

```bash
docker-compose up -d
```

Then:

- **Streamlit:** http://localhost:8501  
- **API:** http://localhost:8000 (docs: http://localhost:8000/docs)  
- **Health:** http://localhost:8000/api/v1/health and http://localhost:8000/api/v1/ready  
- **Create job:** `POST /api/v1/jobs` with multipart form: `files` (1–10 images), `home_value`, optional `zip_code`. Returns `job_id` and `status_url`. Poll `GET /api/v1/jobs/{job_id}` for result.

Set `DATABASE_URL` and `REDIS_URL` when running the API/worker outside Docker (e.g. `export DATABASE_URL=postgresql://repair:repair@localhost:5432/repair_optimizer REDIS_URL=redis://localhost:6379/0`). The Celery worker needs **Standard (2 GB)** or more to run the ML pipeline; otherwise set `FREE_TIER=true` for placeholder results. Optional **`API_KEY`** enables X-API-Key on protected routes (properties, strategies, execute, workspaces, analytics, contractor, export).

## Documentation (Acquisition Package)

- **docs/RENDER.md** – Run on Render: checklist and troubleshooting
- **docs/ARCHITECTURE.md** – Component overview and data flow  
- **docs/API-SPEC.md** – OpenAPI and key endpoints  
- **docs/DATABASE-SCHEMA.md** – Tables and relationships  
- **docs/RUNBOOK.md** – Run, env, health, common issues  
- **docs/SECURITY.md** – API key, encryption, GDPR/CCPA notes  
- **docs/OPENDOOR-INTEGRATION.md** – Integration outline for Opendoor  

## License

MIT
