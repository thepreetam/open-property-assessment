"""
AI Property Condition & Value Assessment – Opendoor-inspired prototype.
Upload 1–5 photos → room classification, YOLO detection, BLIP/LLaVA condition analysis → illustrative price adjustment.
Heavy libs (torch, transformers, cv2, ultralytics) are imported only when needed to keep free-tier memory under 512MB.
"""
import os
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

# ── Flags (no torch import here so cold start stays light) ──
USE_LLAVA = os.environ.get("USE_LLAVA", "false").lower() == "true"
FREE_TIER = os.environ.get("FREE_TIER", "true").lower() == "true"

# ── Cached heavy resources (lazy imports inside to avoid loading torch/transformers until first upload) ──
@st.cache_resource(show_spinner="Loading room classifier...")
def load_room_classifier():
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    # andupets is compatible with transformers image-classification pipeline (JuanMa360 is not in newer transformers)
    return pipeline(
        "image-classification",
        model="andupets/real-estate-image-classification",
        device=device,
    )


@st.cache_resource(show_spinner="Loading BLIP captioner...")
def load_captioner():
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    # transformers 5.x uses "image-text-to-text" instead of "image-to-text"
    return pipeline("image-text-to-text", model="Salesforce/blip-image-captioning-large", device=device)


@st.cache_resource(show_spinner="Loading YOLO detector...")
def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


def _load_llava():
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", low_cpu_mem_usage=True
    )
    return processor, model


@st.cache_resource(show_spinner="Loading LLaVA (condition analysis)...")
def load_llava():
    if not USE_LLAVA:
        return None
    return _load_llava()


def get_condition_description(image, room: str, captioner_pipeline, llava_processor_model):
    """Use LLaVA if available and successful, else BLIP caption. Returns placeholder if no captioner (free tier)."""
    if captioner_pipeline is None:
        return "(Condition analysis and object detection need more memory. Use a paid Render instance for full analysis.)"
    if llava_processor_model is not None:
        try:
            import torch
            processor, model = llava_processor_model
            prompt = (
                f"Assess this {room} on quality scale 1–10. "
                "Detail upgrades (modern appliances, premium materials), wear/outdated features, "
                "needed repairs (cracks, damage, mold, outdated fixtures). Be objective, specific, and concise."
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=220, do_sample=False)
            return processor.decode(output[0], skip_special_tokens=True).strip()
        except Exception:
            pass
    caption_result = captioner_pipeline(image, max_new_tokens=80)
    if isinstance(caption_result, list) and caption_result:
        text = caption_result[0].get("generated_text", caption_result[0]) if isinstance(caption_result[0], dict) else str(caption_result[0])
        return text
    return ""


def _count_appliances(yolo_results):
    """Count COCO appliance-like classes across all boxes (iterate per box)."""
    appliance_names = {"refrigerator", "oven", "microwave", "sink", "toaster"}
    count = 0
    for r in yolo_results:
        if r.boxes is None:
            continue
        for i in range(len(r.boxes.cls)):
            cls_id = int(r.boxes.cls[i])
            name = r.names.get(cls_id, "")
            if name in appliance_names:
                count += 1
    return count


def compute_adjustment(room: str, desc: str, yolo_results, base_home_value: int):
    """Heuristic price adjustment from room type, description text, and YOLO signals."""
    desc_lower = (desc or "").lower()
    adj_pct = 0.0
    notes = []

    if "kitchen" in room.lower():
        if any(w in desc_lower for w in ["modern", "stainless", "granite", "quartz", "new", "upgraded", "premium"]):
            adj_pct += 0.06
            notes.append("Premium kitchen upgrades")
        if any(w in desc_lower for w in ["outdated", "old", "worn", "laminate", "formica"]):
            adj_pct -= 0.04
            notes.append("Outdated kitchen")
        if any(w in desc_lower for w in ["crack", "peel", "damage", "repair"]):
            adj_pct -= 0.03
            notes.append("Repair needs detected")

    elif "bathroom" in room.lower():
        if "modern" in desc_lower or ("tile" in desc_lower and "new" in desc_lower):
            adj_pct += 0.05
            notes.append("Modern bathroom")
        else:
            adj_pct -= 0.03
            notes.append("Bathroom not notably modern")

    elif any(x in room.lower() for x in ["exterior", "yard", "facade"]):
        if "well-maintained" in desc_lower or "clean" in desc_lower:
            adj_pct += 0.035
            notes.append("Well-maintained exterior")
        if any(w in desc_lower for w in ["damage", "overgrown", "crack", "peel"]):
            adj_pct -= 0.045
            notes.append("Exterior wear/damage")

    appliance_count = _count_appliances(yolo_results)
    if appliance_count >= 3:
        adj_pct += 0.015
        notes.append("High appliance count → functional/modern")

    dollar_adj = base_home_value * adj_pct
    return round(dollar_adj), round(adj_pct * 100, 1), notes


def annotate_image(image: Image.Image, yolo_results, desc: str) -> Image.Image:
    """Draw YOLO boxes on image; color by keyword heuristic (green / orange / red). Lazy-import cv2."""
    import cv2
    img_arr = np.array(image)
    if img_arr.ndim == 2:
        img_cv = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)
    else:
        img_cv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    desc_lower = (desc or "").lower()

    for r in yolo_results:
        if r.boxes is None:
            continue
        for i in range(len(r.boxes.xyxy)):
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])
            cls_id = int(r.boxes.cls[i])
            cls_name = r.names.get(cls_id, "object")
            conf = float(r.boxes.conf[i])
            label = f"{cls_name} {conf:.2f}"

            color = (0, 255, 0)
            if any(w in cls_name.lower() for w in ["crack", "damage"]):
                color = (0, 0, 255)
            elif any(w in desc_lower for w in ["outdated", "old", "worn"]):
                color = (0, 165, 255)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_cv, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


# ── App layout (models load lazily on first upload to keep page load fast) ──
st.set_page_config(page_title="AI Property Assessment", layout="wide")
st.title("AI Property Condition & Value Assessment")
st.caption(
    "Opendoor-inspired prototype: Upload photos → AI analyzes room quality, features, repairs → suggests illustrative adjustments."
)

with st.sidebar:
    st.header("Settings")
    st.info(("Free tier (room only)" if FREE_TIER else ("LLaVA" if USE_LLAVA else "BLIP + YOLO")))
    st.caption("Models load when you upload photos." + (" Free/Starter (512MB) can't run models – upgrade to Standard (2GB) and set FREE_TIER=false." if FREE_TIER else " First run may take 2–5 min."))
    home_value = st.slider("Estimated home value ($)", 200_000, 2_000_000, 500_000, step=50_000)
    st.markdown("Adjustments are **illustrative** (±%) based on detected condition.")
    st.markdown("**Disclaimer**: Not financial advice. Inspired by Opendoor's AI assessments/RiskAI.")

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
                st.image(image, use_column_width=True, caption="Uploaded – upgrade for analysis")
            st.markdown("---")
        with tab2:
            st.write("Upload photos and upgrade to Standard (2 GB) or higher to see analysis and summary here.")
    else:
        with st.spinner("Loading AI models (first time may take 2–5 min)…"):
            room_classifier = load_room_classifier()
            captioner = load_captioner()
            yolo_model = load_yolo()
            llava_processor_model = load_llava() if USE_LLAVA else None

        tab1, tab2 = st.tabs(["Per-Photo Details", "Aggregate Summary"])
        all_data = []
        progress = st.progress(0.0)

        for idx, file in enumerate(uploaded_files):
            progress.progress((idx + 1) / len(uploaded_files))
            image = Image.open(file).convert("RGB")

            with tab1:
                st.subheader(f"Photo {idx + 1}")
                col1, col2 = st.columns(2)
                col1.image(image, use_column_width=True, caption="Original")

                room_results = room_classifier(image)
                room = room_results[0]["label"]
                room_conf = room_results[0]["score"]
                col2.metric("Detected Room", room, f"{room_conf:.1%} conf")

                desc = get_condition_description(image, room, captioner, llava_processor_model)
                col2.markdown(f"**Condition analysis**\n{desc[:500]}" + ("…" if len(desc) > 500 else ""))

                yolo_results = yolo_model(image, verbose=False)
                annotated = annotate_image(image.copy(), yolo_results, desc)
                col1.image(annotated, caption="YOLO detection + highlights", use_column_width=True)

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
