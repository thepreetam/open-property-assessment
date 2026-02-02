# AI Property Condition & Value Assessment

Opendoor-inspired prototype: upload 1–5 home photos (kitchen, bathroom, living room, exterior) and get room classification, object detection, condition analysis, and an **illustrative** price adjustment suggestion.

## Features

- **Room classification** – Detects room type (kitchen, bathroom, bedroom, exterior, etc.) with confidence
- **Object detection** – YOLO bounding boxes on appliances, furniture, fixtures
- **Condition analysis** – BLIP or LLaVA description → keyword-based quality/repair signals
- **Price adjustment** – Heuristic ±$ and % based on detected upgrades vs. wear/repairs
- **Multi-photo** – Aggregate results across uploads; per-photo details and summary table with CSV export

## Deploy on Render

**Recommended:** Run the app on Render so it’s always available at a public URL.

1. Push this repo to GitHub (or GitLab/Bitbucket).
2. In the [Render Dashboard](https://dashboard.render.com), click **New → Web Service**, connect the repo.
3. Use these settings (or use the repo’s `render.yaml` Blueprint for one-click deploy):
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
   - **Python:** Render uses `runtime.txt` (e.g. `python-3.11.7`) if present.
4. **Free tier (512MB):** The app runs in **room-only mode** (`FREE_TIER=true` in `render.yaml`) so it fits in 512MB: only room classification runs; condition text and object detection show a message that a paid instance is needed. Spins down after 15 min idle (cold start ~1 min).
5. **Paid instance (1GB+ RAM):** Set env var **`FREE_TIER=false`** to enable full analysis (BLIP captioning + YOLO detection). Optionally set **`USE_LLAVA=true`** for LLaVA condition text (use 2GB+ RAM).

## Opendoor tie-in

This prototype automates condition flagging for faster assessments and illustrates how CV + multimodal signals could refine instant-offer pricing (e.g. RiskAI-style). **Not** a production valuation tool.

## Limitations

- Adjustments are **rule-based heuristics**, not accurate valuations. Not financial advice.
- Bias: lighting/angle and training data affect results; production would need diverse datasets.
- On Render free tier only the room classifier runs (512MB limit); set `FREE_TIER=false` on a paid instance for BLIP + YOLO.

## Run locally (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

First run loads models (1–2 min on CPU). Use the sidebar to set estimated home value; adjustments scale with it.

## License

MIT
