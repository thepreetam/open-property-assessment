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
4. **Free and Starter (512 MB RAM):** Both have 512 MB, which is **not enough** to run the AI models. The app will stay up and show an upgrade message when you upload photos. Spins down after 15 min idle (cold start ~1 min).
5. **Standard (2 GB) or higher:** Upgrade in Render → Settings → Instance type (e.g. **Standard $25/mo, 2 GB**). Set env var **`FREE_TIER=false`** to enable room classification, BLIP captioning, and YOLO detection. Optionally **`USE_LLAVA=true`** for LLaVA (use Standard or higher).
6. **Optional – Hugging Face token:** To remove the "unauthenticated requests" warning and get higher rate limits and faster model downloads, add env var **`HF_TOKEN`** (or `HUGGING_FACE_HUB_TOKEN`) in the Render service with a [Hugging Face read token](https://huggingface.co/settings/tokens). Create a token at https://huggingface.co/settings/tokens (read is enough).

## Opendoor tie-in

This prototype automates condition flagging for faster assessments and illustrates how CV + multimodal signals could refine instant-offer pricing (e.g. RiskAI-style). **Not** a production valuation tool.

## Limitations

- Adjustments are **rule-based heuristics**, not accurate valuations. Not financial advice.
- Bias: lighting/angle and training data affect results; production would need diverse datasets.
- Free and Starter (512 MB) can't run the models; the app shows an upgrade message. Use Standard (2 GB) or higher and set `FREE_TIER=false` for full analysis.

## Run locally (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

First run loads models (1–2 min on CPU). Use the sidebar to set estimated home value; adjustments scale with it.

## License

MIT
