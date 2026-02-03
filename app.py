"""
Repair ROI Optimizer – Opendoor-inspired prototype.
Upload 1–5 photos → room classification, YOLO, BLIP/LLaVA condition analysis → illustrative price adjustment + three repair strategies (cost, uplift, ROI).
Uses shared pipeline from core/; heavy libs load lazily on first upload.
"""
import os
import streamlit as st
from PIL import Image
import pandas as pd

from core.pipeline import (
    load_room_classifier as _load_room_classifier,
    load_captioner as _load_captioner,
    load_yolo as _load_yolo,
    load_llava as _load_llava_core,
    get_condition_description,
    compute_adjustment,
    annotate_image,
)
from core.repair_engine import get_findings, generate_strategies

# ── Flags (no torch import here so cold start stays light) ──
USE_LLAVA = os.environ.get("USE_LLAVA", "false").lower() == "true"
FREE_TIER = os.environ.get("FREE_TIER", "true").lower() == "true"

# ── Cached heavy resources (delegate to core pipeline; Streamlit caches for session) ──
@st.cache_resource(show_spinner="Loading room classifier...")
def load_room_classifier():
    return _load_room_classifier()


@st.cache_resource(show_spinner="Loading BLIP captioner...")
def load_captioner():
    return _load_captioner()


@st.cache_resource(show_spinner="Loading YOLO detector...")
def load_yolo():
    return _load_yolo()


@st.cache_resource(show_spinner="Loading LLaVA (condition analysis)...")
def load_llava():
    if not USE_LLAVA:
        return None
    return _load_llava_core(use_llava=True)


# ── App layout (models load lazily on first upload to keep page load fast) ──
st.set_page_config(page_title="Repair ROI Optimizer", layout="wide")
st.title("Repair ROI Optimizer")
st.caption(
    "Turn findings into profit-maximizing repair options. Upload photos + set value and zip → get 3 strategies with cost, uplift, and ROI."
)

with st.sidebar:
    st.header("Settings")
    st.info(("Free tier (room only)" if FREE_TIER else ("LLaVA" if USE_LLAVA else "BLIP + YOLO")))
    st.caption("Models load when you upload photos." + (" Free/Starter (512MB) can't run models – upgrade to Standard (2GB) and set FREE_TIER=false." if FREE_TIER else " First run may take 2–5 min."))
    home_value = st.slider("Estimated home value ($)", 200_000, 2_000_000, 500_000, step=50_000)
    zip_code = st.text_input("Zip code (for market context)", value="", placeholder="e.g. 85001")
    st.markdown("Adjustments and repair strategies are **illustrative** (not financial advice).")
    st.markdown("**Disclaimer**: Inspired by Opendoor's Repair Co-Pilot and RiskAI.")

uploaded_files = st.file_uploader(
    "Upload 1–5 photos (kitchen, bath, exterior...)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    # Free tier (512MB): don't load any models – avoids OOM. Starter is also 512MB; need Standard (2GB)+ for analysis.
    if FREE_TIER:
        tab1, tab2 = st.tabs(["Per-Photo Details", "Aggregate Summary"])
        st.info(
            "**Photo analysis needs more RAM.** Free and Starter instances have 512 MB, which isn't enough for the AI models. "
            "Upgrade to **Standard (2 GB)** or higher in Render → Settings → Instance type, then set **FREE_TIER=false** in Environment to enable room classification and full analysis."
        )
        for idx, file in enumerate(uploaded_files):
            image = Image.open(file).convert("RGB")
            with tab1:
                st.subheader(f"Photo {idx + 1}")
                st.image(image, caption="Uploaded – upgrade for analysis")
            st.markdown("---")
        with tab2:
            st.write("Upload photos and upgrade to Standard (2 GB) or higher to see analysis and summary here.")
    else:
        with st.spinner("Loading AI models (first time may take 2–5 min)…"):
            room_classifier = load_room_classifier()
            captioner = load_captioner()
            yolo_model = load_yolo()
            llava_processor_model = load_llava() if USE_LLAVA else None

        tab1, tab2, tab3 = st.tabs(["Per-Photo Details", "Aggregate Summary", "Repair Strategies"])
        all_data = []
        progress = st.progress(0.0)

        for idx, file in enumerate(uploaded_files):
            progress.progress((idx + 1) / len(uploaded_files))
            image = Image.open(file).convert("RGB")

            with tab1:
                st.subheader(f"Photo {idx + 1}")
                col1, col2 = st.columns(2)
                col1.image(image, caption="Original")

                room_results = room_classifier(image)
                room = room_results[0]["label"]
                room_conf = room_results[0]["score"]
                col2.metric("Detected Room", room, f"{room_conf:.1%} conf")

                desc = get_condition_description(image, room, captioner, llava_processor_model)
                col2.markdown(f"**Condition analysis**\n{desc[:500]}" + ("…" if len(desc) > 500 else ""))

                yolo_results = yolo_model(image, verbose=False)
                annotated = annotate_image(image.copy(), yolo_results, desc)
                col1.image(annotated, caption="YOLO detection + highlights")

                adj_dollar, adj_pct, notes = compute_adjustment(room, desc, yolo_results, home_value)
                all_data.append({
                    "Photo": idx + 1,
                    "Room": room,
                    "Quality excerpt": (desc.split("\n")[0] if "\n" in desc else desc)[:60],
                    "Adjustment $": adj_dollar,
                    "% Adj": adj_pct,
                    "Notes": "; ".join(notes),
                })
                if adj_dollar > 0:
                    st.success(f"Suggested adjustment: **+${adj_dollar:,}** ({adj_pct:+.1f}%) – illustrative only")
                elif adj_dollar < 0:
                    st.warning(f"Suggested adjustment: **${adj_dollar:,}** ({adj_pct:.1f}%) – illustrative only")
                else:
                    st.info("Neutral adjustment – add more logic or photos.")
                st.markdown("---")

        progress.empty()

        # Repair Optimizer: findings → 3 strategies + recommendation (shared core.repair_engine)
        findings = get_findings(all_data)
        result = generate_strategies(findings, home_value, zip_code or None)
        strategies = result["strategies"]
        recommendation = result["recommendation"]

        with tab3:
            if not findings:
                st.info(
                    "No repair findings from these photos. Upload photos showing visible issues (e.g. outdated kitchen, "
                    "dated bathroom, exterior wear) to see Budget / Value / Premium repair strategies with ROI."
                )
            else:
                st.subheader("Findings")
                st.write("Issues detected that have repair options: " + ", ".join(f["category"].replace("_", " ") for f in findings))
                st.subheader("Repair strategies")
                c1, c2, c3 = st.columns(3)
                for col, (key, s) in zip([c1, c2, c3], strategies.items()):
                    with col:
                        st.metric(s["name"], f"${s['cost']:,}", f"{s['roi_pct']}% ROI")
                        st.caption(s["philosophy"])
                        st.write(f"**Uplift:** +${s['uplift']:,} · **Timeline:** {s['timeline_days']} days")
                st.success(f"**Recommendation:** {recommendation['strategy_name']} – {recommendation['reason']}")

        with tab2:
            if all_data:
                df = pd.DataFrame(all_data)
                st.dataframe(df, use_container_width=True)
                total_adj = df["Adjustment $"].sum()
                avg_pct = df["% Adj"].mean()

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Suggested Adjustment", f"${total_adj:,.0f}", delta_color="normal")
                c2.metric("Avg % Adjustment", f"{avg_pct:.1f}%")
                c3.metric("Photos Analyzed", len(uploaded_files))

                if total_adj > 0:
                    st.success("Overall positive uplift likely from detected upgrades & condition.")
                elif total_adj < 0:
                    st.warning("Potential deductions from wear, outdated features, or repairs.")
                else:
                    st.info("Neutral net adjustment – condition appears average.")

                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Summary CSV", csv_bytes, "property_assessment_summary.csv", "text/csv")

st.markdown(
    """
**How this could help at Opendoor**  
Automates condition flagging for faster Key Agent assessments, refines instant-offer pricing via RiskAI-like signals.  
Prototype only – production would use fine-tuned models + diverse training data.
"""
)
