# Run on Render

This app is set up to run as a **Web Service** on [Render](https://render.com).

## One-click deploy (Blueprint)

1. Push this repo to GitHub (or connect GitLab/Bitbucket).
2. In Render Dashboard: **New → Blueprint**, select the repo. Render will read `render.yaml` and create the **property-assessment** Web Service.
3. Or **New → Web Service**, connect the repo, and use the settings below.

## Required settings

| Setting | Value |
|--------|--------|
| **Build command** | `pip install -r requirements.txt` |
| **Start command** | `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0` |
| **Python** | Use `runtime.txt` in the repo (e.g. `python-3.11.7`). |

Render sets **`PORT`** automatically; the start command must use `--server.port=$PORT` and `--server.address=0.0.0.0` so the app listens on the correct interface.

## Instance type and env vars

- **Free / Starter (512 MB):** Not enough RAM for the AI models. The app will start; when you upload photos it shows an upgrade message. Set **`FREE_TIER=true`** (default) so no models load. Spins down after ~15 min idle (cold start ~1 min).
- **Standard (2 GB) or higher:** Required for full analysis. In Render → **Settings → Instance type**, choose **Standard**. Set **`FREE_TIER=false`** in **Environment** to enable room classification, BLIP, YOLO, and Repair Strategies.
- **Optional:** **`HF_TOKEN`** (Hugging Face read token) for faster model downloads and higher rate limits.
- **Optional:** **`USE_LLAVA=true`** for LLaVA condition analysis (use Standard or more RAM).

## Checklist

- [ ] Repo connected; build and start commands as above.
- [ ] `PORT` is not set manually (Render injects it).
- [ ] For full ML: Instance type **Standard (2 GB)** and **`FREE_TIER=false`**.
- [ ] After deploy, open the service URL; first load (with models) may take 2–5 min.

## Troubleshooting

- **"No open ports detected"** – Ensure start command uses `--server.port=$PORT --server.address=0.0.0.0`. Streamlit binds after startup; if the app loads models, the first request can be slow.
- **Out of memory (512 MB)** – Upgrade to Standard (2 GB) and set `FREE_TIER=false`, or keep Free and leave `FREE_TIER=true` (placeholder mode).
- **Cold start** – Free tier spins down after idle; first request after that can take ~1 min.
