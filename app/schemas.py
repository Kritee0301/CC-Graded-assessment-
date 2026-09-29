"""Pydantic request/response schemas for the customer segmentation API.

These models define the public contract of the service. FastAPI validates
every request body against them, so out-of-range or misspelled fields are
rejected with HTTP 422 before any model code runs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Canonical feature order.
# This MUST match FEATURES in train_model.py: the scaler and KMeans centroids
# were fitted on exactly this order, and to_feature_vector() relies on it.
# ---------------------------------------------------------------------------
FEATURE_ORDER: tuple[str, ...] = (
    "age",
    "annual_income_k",
    "spending_score",
    "purchase_frequency",
    "avg_order_value",
)

# Mirrors FEATURE_BOUNDS in train_model.py. Requests outside these ranges are
# rejected rather than silently extrapolated beyond the training data.
AGE_MIN, AGE_MAX = 18, 75
INCOME_MIN, INCOME_MAX = 15.0, 200.0
SPEND_MIN, SPEND_MAX = 1.0, 100.0
FREQ_MIN, FREQ_MAX = 1.0, 30.0
AOV_MIN, AOV_MAX = 10.0, 400.0

class CustomerFeatures(BaseModel):
    """Behavioural features for a single customer."""

    model_config = ConfigDict(
        extra="forbid",  # typo'd or unknown fields are errors, not ignored
        json_schema_extra={
            "example": {
                "age": 40,
                "annual_income_k": 61.0,
                "spending_score": 51,
                "purchase_frequency": 13,
                "avg_order_value": 74.0,
            }
        },
    )

    age: int = Field(
        ...,
        ge=AGE_MIN,
        le=AGE_MAX,
        description="Customer age in years.",
    )
    annual_income_k: float = Field(
        ...,
        ge=INCOME_MIN,
        le=INCOME_MAX,
        description="Annual income in thousands of USD.",
    )
    spending_score: float = Field(
        ...,
        ge=SPEND_MIN,
        le=SPEND_MAX,
        description="Behavioural spending score (1 = lowest, 100 = highest).",
    )
    purchase_frequency: float = Field(
        ...,
        ge=FREQ_MIN,
        le=FREQ_MAX,
        description="Number of purchases per year.",
    )
    avg_order_value: float = Field(
        ...,
        ge=AOV_MIN,
        le=AOV_MAX,
        description="Average order value in USD.",
    )

    def to_feature_vector(self) -> List[float]:
        """Return the values as a plain list, in FEATURE_ORDER.

        The model expects this exact ordering; building the list here (rather
        than in the route handler) keeps the contract in one place.
        """
        return [float(getattr(self, name)) for name in FEATURE_ORDER]

class PredictionResponse(BaseModel):
    """Prediction for a single customer."""

    segment_id: int = Field(..., ge=0, description="Segment id (0 = highest income).")
    segment_name: str = Field(..., description="Human-readable segment name.")
    segment_description: str = Field(..., description="One-line segment profile.")
    distance_to_centroid: float = Field(
        ...,
        ge=0.0,
        description="Euclidean distance to the assigned centroid, in standardised space.",
    )
    confidence_margin: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Relative separation from the second-closest segment: "
            "0 = sitting on a boundary, 1 = unambiguous. NOT a probability."
        ),
    )
    segment_distances: Dict[str, float] = Field(
        ...,
        description="Distance from this customer to every segment centroid.",
    )

class BatchPredictRequest(BaseModel):
    """Batch prediction request."""

    model_config = ConfigDict(extra="forbid")

    customers: List[CustomerFeatures] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Between 1 and 100 customers per request.",
    )

class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    count: int = Field(..., ge=0, description="Number of predictions returned.")
    predictions: List[PredictionResponse] = Field(...)

class SegmentInfo(BaseModel):
    """One customer segment, as learned during training."""

    segment_id: int = Field(..., ge=0)
    name: str
    description: str
    customer_count: int = Field(
        ..., ge=0, description="Customers from the training set in this segment."
    )
    centroid: Dict[str, float] = Field(
        ..., description="Segment centroid in original feature units."
    )

class SegmentsResponse(BaseModel):
    """All segments known to the model."""

    n_clusters: int = Field(..., ge=1)
    segments: List[SegmentInfo] = Field(...)

class ModelInfoResponse(BaseModel):
    """Model metadata: what it is, how it was trained, how well it scored."""

    model_type: str = Field(..., description="Estimator class name, e.g. 'KMeans'.")
    n_clusters: int = Field(..., ge=1)
    features: List[str] = Field(..., description="Feature names, in model order.")
    feature_bounds: Dict[str, Dict[str, float]] = Field(
        ..., description="Observed min/max per feature in the training data."
    )
    metrics: Dict[str, float] = Field(
        ..., description="Training metrics (inertia, silhouette, adjusted_rand_index)."
    )
    training: Dict[str, Any] = Field(
        ..., description="Training provenance: sample count, seed, versions, timestamp."
    )
    segments: List[SegmentInfo] = Field(...)

class HealthResponse(BaseModel):
    """Liveness/readiness payload. Used by the deploy step and by CI smoke tests."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_path: str = Field(
        ..., description="Model path relative to the project root (no absolute paths)."
    )
    sklearn_version: str
    n_clusters: int
    features: List[str]

class ErrorResponse(BaseModel):
    """Error envelope used for documented error responses."""

    detail: str