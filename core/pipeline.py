"""
ML pipeline: photos → room, condition, YOLO, adjustments → per-photo data + repair strategies.
Callable from both Streamlit (with optional pre-loaded models) and Celery worker (loads models internally).
No Streamlit dependencies; heavy libs imported inside functions for lazy loading.
"""
import os
from typing import Any, Dict, List, Optional

# Lazy imports for torch/transformers/cv2/ultralytics inside functions


def load_room_classifier():
    """Load room classifier (no Streamlit cache)."""
    import torch
    from transformers import pipeline, AutoImageProcessor
    device = 0 if torch.cuda.is_available() else -1
    model_id = "andupets/real-estate-image-classification"
    image_processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
    return pipeline(
        "image-classification",
        model=model_id,
        image_processor=image_processor,
        device=device,
    )


def load_captioner():
    """Load BLIP captioner."""
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("image-text-to-text", model="Salesforce/blip-image-captioning-large", device=device)


def load_yolo():
    """Load YOLO detector."""
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


def load_llava(use_llava: bool = False):
    """Load LLaVA if USE_LLAVA is true."""
    if not use_llava:
        return None
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", low_cpu_mem_usage=True
    )
    return (processor, model)


def get_condition_description(image, room: str, captioner_pipeline, llava_processor_model):
    """Use LLaVA if available and successful, else BLIP caption. Placeholder if no captioner (free tier)."""
    if captioner_pipeline is None:
        return "(Condition analysis needs more memory. Use Standard (2GB) and FREE_TIER=false for full analysis.)"
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
    # image-text-to-text pipeline requires both image and text (prompt)
    prompt = f"Describe the condition and quality of this {room}. Focus on upgrades, wear, and repair needs."
    caption_result = captioner_pipeline(image, text=prompt, max_new_tokens=80)
    if isinstance(caption_result, list) and caption_result:
        text = caption_result[0].get("generated_text", caption_result[0]) if isinstance(caption_result[0], dict) else str(caption_result[0])
        return text
    return ""


def _count_appliances(yolo_results):
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

    if any(w in desc_lower for w in ["floor", "flooring", "carpet", "hardwood"]) and any(w in desc_lower for w in ["worn", "damage", "outdated", "stain"]):
        adj_pct -= 0.02
        notes.append("Worn flooring")

    appliance_count = _count_appliances(yolo_results)
    if appliance_count >= 3:
        adj_pct += 0.015
        notes.append("High appliance count → functional/modern")

    dollar_adj = base_home_value * adj_pct
    return round(dollar_adj), round(adj_pct * 100, 1), notes


def annotate_image(image, yolo_results, desc: str):
    """Draw YOLO boxes on image; color by keyword heuristic."""
    import cv2
    import numpy as np
    from PIL import Image
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


def run_pipeline(
    images: List[Any],
    home_value: int,
    zip_code: Optional[str] = None,
    free_tier: bool = True,
    use_llava: bool = False,
    room_classifier=None,
    captioner=None,
    yolo_model=None,
    llava_processor_model=None,
) -> Dict[str, Any]:
    """
    Run full pipeline on a list of PIL Images.
    Returns dict: all_data (list of per-photo rows), strategies, recommendation.
    If free_tier=True, skips loading ML models and returns placeholder all_data and empty strategies.
    Optional model args allow caller to pass pre-loaded models (e.g. from Streamlit cache).
    """
    from core.repair_engine import get_findings, generate_strategies

    if free_tier:
        all_data = []
        for idx, image in enumerate(images):
            all_data.append({
                "Photo": idx + 1,
                "Room": "unknown",
                "Quality excerpt": "(Free tier – upgrade for analysis)",
                "Adjustment $": 0,
                "% Adj": 0.0,
                "Notes": "",
            })
        return {
            "all_data": all_data,
            "strategies": {},
            "recommendation": {"strategy_name": "N/A", "reason": "Enable full analysis (FREE_TIER=false) for repair strategies."},
        }

    # Load models if not provided
    if room_classifier is None:
        room_classifier = load_room_classifier()
    if captioner is None:
        captioner = load_captioner()
    if yolo_model is None:
        yolo_model = load_yolo()
    if llava_processor_model is None:
        llava_processor_model = load_llava(use_llava=use_llava)

    from PIL import Image
    import io
    all_data = []
    for idx, img_in in enumerate(images):
        if hasattr(img_in, "convert"):
            image = img_in.convert("RGB")
        elif isinstance(img_in, bytes):
            image = Image.open(io.BytesIO(img_in)).convert("RGB")
        elif hasattr(img_in, "read"):
            image = Image.open(img_in).convert("RGB")
        else:
            image = img_in

        room_results = room_classifier(image)
        room = room_results[0]["label"]
        desc = get_condition_description(image, room, captioner, llava_processor_model)
        yolo_results = yolo_model(image, verbose=False)
        adj_dollar, adj_pct, notes = compute_adjustment(room, desc, yolo_results, home_value)

        all_data.append({
            "Photo": idx + 1,
            "Room": room,
            "Quality excerpt": (desc.split("\n")[0] if "\n" in desc else desc)[:60],
            "Adjustment $": adj_dollar,
            "% Adj": adj_pct,
            "Notes": "; ".join(notes),
        })

    findings = get_findings(all_data)
    result = generate_strategies(findings, home_value, zip_code)
    return {
        "all_data": all_data,
        "strategies": result["strategies"],
        "recommendation": result["recommendation"],
    }
