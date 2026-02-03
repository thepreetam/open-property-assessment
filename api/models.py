"""
Pydantic request/response models for the API.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request body for POST /api/v1/jobs (analyze)."""

    home_value: int = Field(..., ge=200_000, le=10_000_000, description="Estimated home value in USD")
    zip_code: Optional[str] = Field(None, max_length=20)
    # Photos are sent as multipart/form-data; this model is used for JSON body or query params
    # When using form + files, the route will accept files separately.


class JobCreateResponse(BaseModel):
    """Response after creating an analysis job."""

    job_id: str
    status: str = "pending"
    status_url: str
    estimated_completion: str = "2 minutes"


class JobStatusResponse(BaseModel):
    """Job status and result."""

    job_id: str
    status: str  # pending | processing | completed | failed
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BulkJobItem(BaseModel):
    """One row for bulk job creation."""

    address: Optional[str] = None
    zip_code: Optional[str] = None
    home_value: int = Field(..., ge=200_000, le=10_000_000)
    photo_urls: List[str] = Field(..., min_length=1, max_length=10, description="1–10 image URLs")


class BulkJobRequest(BaseModel):
    """Request body for POST /api/v1/jobs/bulk."""

    jobs: List[BulkJobItem] = Field(..., max_length=100, description="Up to 100 jobs per request")


class BulkJobError(BaseModel):
    index: int
    reason: str


class BulkJobResponse(BaseModel):
    job_ids: List[str]
    errors: List[BulkJobError] = []


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"


class ReadinessResponse(BaseModel):
    """Readiness check (DB, Redis optional)."""

    status: str  # ready | not_ready
    database: str = "unknown"
    redis: str = "unknown"


# Phase 2: Properties, Strategies, Execute
class PropertyCreate(BaseModel):
    address: Optional[str] = None
    zip_code: Optional[str] = None
    home_value: Optional[int] = None
    workspace_id: Optional[str] = None


class PropertyResponse(BaseModel):
    id: str
    workspace_id: Optional[str] = None
    address: Optional[str] = None
    zip_code: Optional[str] = None
    home_value: Optional[str] = None
    created_at: Optional[str] = None


class StrategiesResponse(BaseModel):
    job_id: Optional[str] = None
    property_id: Optional[str] = None
    strategies: Dict[str, Any]
    recommendation: Dict[str, Any]
    timeline: Optional[List[Dict[str, Any]]] = None


class ExecuteRequest(BaseModel):
    property_id: str
    strategy_key: str = Field(..., pattern="^(budget_flip|value_add|premium_reno)$")
    job_id: Optional[str] = None


class ExecuteResponse(BaseModel):
    execution_id: str
    status: str = "dispatched"
    property_id: str
    strategy_key: str
    timeline: Optional[Dict[str, Any]] = None
