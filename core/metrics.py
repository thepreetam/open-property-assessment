"""
Prometheus metrics for jobs and pipeline. Set PROMETHEUS_MULTIPROC_DIR in API and worker for multi-process aggregation.
"""
import os
from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.multiprocess import MultiProcessCollector

_multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR", "").strip()

# Use default registry so multiprocess mode writes to PROMETHEUS_MULTIPROC_DIR
jobs_started = Counter(
    "repair_optimizer_jobs_started_total",
    "Total analysis jobs created (enqueued).",
)
jobs_completed = Counter(
    "repair_optimizer_jobs_completed_total",
    "Total analysis jobs completed successfully.",
)
jobs_failed = Counter(
    "repair_optimizer_jobs_failed_total",
    "Total analysis jobs failed.",
)
pipeline_duration_seconds = Histogram(
    "repair_optimizer_pipeline_duration_seconds",
    "Pipeline run duration in seconds.",
)


def metrics_output():
    """Generate Prometheus text format; aggregate from all processes if PROMETHEUS_MULTIPROC_DIR is set."""
    if _multiproc_dir:
        registry = CollectorRegistry()
        MultiProcessCollector(registry, _multiproc_dir)
        return generate_latest(registry), CONTENT_TYPE_LATEST
    from prometheus_client import REGISTRY
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
