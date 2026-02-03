"""
MLflow logging for pipeline runs: model versions, params, and metrics per job.
Optional: set MLFLOW_TRACKING_URI (and MLFLOW_EXPERIMENT_NAME) in env; otherwise no-op or local ./mlruns.
"""
import os
from typing import Any, Dict, Optional


def _is_enabled() -> bool:
    """Use MLflow unless explicitly disabled (MLFLOW_TRACKING_URI=false). Default: local ./mlruns."""
    return os.environ.get("MLFLOW_TRACKING_URI", "x").strip().lower() != "false"


def log_pipeline_run(
    job_id: str,
    zip_code: Optional[str],
    home_value: int,
    result: Dict[str, Any],
    model_variant: str = "blip",
    free_tier: bool = False,
) -> None:
    """
    Log one pipeline invocation to MLflow: params (job_id, zip_code, home_value, model_variant, free_tier)
    and metrics (num_photos, num_findings, strategy_recommended). No-op if MLflow not configured.
    """
    if not _is_enabled():
        return
    try:
        import mlflow
    except ImportError:
        return

    exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "repair_optimizer_pipeline")
    try:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            mlflow.create_experiment(exp_name)
        mlflow.set_experiment(exp_name)
    except Exception:
        pass

    with mlflow.start_run(run_name=f"job_{job_id[:8]}"):
        mlflow.log_param("job_id", job_id)
        mlflow.log_param("zip_code", zip_code or "")
        mlflow.log_param("home_value", home_value)
        mlflow.log_param("model_variant", model_variant)
        mlflow.log_param("free_tier", str(free_tier).lower())

        all_data = result.get("all_data") or []
        mlflow.log_metric("num_photos", len(all_data))

        rec = result.get("recommendation") or {}
        strategy_name = rec.get("strategy_name") or "N/A"
        mlflow.log_param("strategy_recommended", strategy_name)

        # Heuristic: findings ~ non-empty notes across photos
        num_findings = sum(1 for row in all_data if (row.get("Notes") or "").strip())
        mlflow.log_metric("num_findings", num_findings)

        # Optional: log model "versions" as params (we don't load package versions in worker for speed)
        mlflow.log_param("room_classifier", "andupets/real-estate-image-classification")
        mlflow.log_param("captioner", "Salesforce/blip-image-captioning-large")
        mlflow.log_param("yolo", "yolov8n.pt")
