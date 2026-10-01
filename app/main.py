# """FastAPI application exposing the trained customer segmentation model.

# Endpoints:
#     GET  /              service index
#     GET  /health        readiness + model metadata
#     POST /predict       segment a single customer
#     POST /predict/batch segment up to 100 customers in one call
#     GET  /segments      all segments with centroids and sizes
#     GET  /model/info    model provenance and training metrics

# The model is loaded once during application startup (lifespan). If the
# artifact is missing or malformed the process exits immediately with a clear
# message instead of serving a broken service.
# """

# from __future__ import annotations

# import logging
# import os
# from contextlib import asynccontextmanager
# from typing import Any, AsyncIterator, Dict, List

# from fastapi import Depends, FastAPI, HTTPException, Request

# from app.model import ModelLoadError, SegmentationModel
# from app.schemas import (
#     BatchPredictionResponse,
#     BatchPredictRequest,
#     CustomerFeatures,
#     ErrorResponse,
#     HealthResponse,
#     ModelInfoResponse,
#     PredictionResponse,
#     SegmentsResponse,
# )

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s %(name)s: %(message)s",
# )
# logger = logging.getLogger("app.main")

# SERVICE_NAME = "customer-segmentation-api"
# SERVICE_VERSION = "1.0.0"

# # Optional override, e.g. MODEL_PATH=/models/other.pkl under Docker or in CI.
# MODEL_PATH_ENV = "MODEL_PATH"

# # Where to document an error we raise ourselves. FastAPI's built-in 422 body
# # for malformed requests is left as-is (it is the standard shape).
# ERROR_RESPONSES: Dict[int | str, Dict[str, Any]] = {
#     400: {"model": ErrorResponse, "description": "Feature values could not be used."},
#     503: {"model": ErrorResponse, "description": "Model is not loaded."},
# }

# @asynccontextmanager
# async def lifespan(app: FastAPI) -> AsyncIterator[None]:
#     """Load the model once at startup; fail fast if it cannot be loaded."""
#     model_path = os.getenv(MODEL_PATH_ENV) or None
#     logger.info(
#         "Starting %s v%s (model path: %s)",
#         SERVICE_NAME,
#         SERVICE_VERSION,
#         model_path or "default",
#     )

#     try:
#         model = SegmentationModel.load(model_path)
#     except ModelLoadError as exc:
#         # Re-raised on purpose: crash now, not on the first user request.
#         logger.critical("Startup aborted: %s", exc)
#         raise

#     app.state.model = model
#     logger.info(
#         "Ready: %s | %d clusters | features=%s",
#         type(model.model).__name__,
#         model.n_clusters,
#         ",".join(model.feature_names),
#     )
#     try:
#         yield
#     finally:
#         app.state.model = None
#         logger.info("Shutdown complete.")

# app = FastAPI(
#     title="Customer Segmentation API",
#     description=(
#         "KMeans-based customer segmentation served over HTTP. "
#         "Customers are described by five behavioural features and assigned to "
#         "one of four segments. Trained offline by train_model.py; the fitted "
#         "scaler and model are loaded from a single joblib artifact at startup."
#     ),
#     version=SERVICE_VERSION,
#     lifespan=lifespan,
#     contact={"name": "Customer Segmentation API"},
# )

# def get_model(request: Request) -> SegmentationModel:
#     """Dependency that returns the loaded model, or 503 if unavailable."""
#     model = getattr(request.app.state, "model", None)
#     if model is None:  # pragma: no cover - only reachable if lifespan is bypassed
#         raise HTTPException(status_code=503, detail="Model is not loaded.")
#     return model

# @app.get("/", tags=["meta"], summary="Service index")
# def root() -> Dict[str, Any]:
#     """Identify the service and list its endpoints."""
#     return {
#         "service": SERVICE_NAME,
#         "version": SERVICE_VERSION,
#         "status": "running",
#         "endpoints": [
#             "GET  /",
#             "GET  /health",
#             "POST /predict",
#             "POST /predict/batch",
#             "GET  /segments",
#             "GET  /model/info",
#             "GET  /docs",
#             "GET  /openapi.json",
#         ],
#     }

# @app.get(
#     "/health",
#     tags=["meta"],
#     summary="Readiness probe",
#     response_model=HealthResponse,
#     responses=ERROR_RESPONSES,
# )
# def health(model: SegmentationModel = Depends(get_model)) -> HealthResponse:
#     """Report readiness plus which model artifact is loaded."""
#     return HealthResponse.model_validate(model.health())

# @app.post(
#     "/predict",
#     tags=["prediction"],
#     summary="Segment a single customer",
#     response_model=PredictionResponse,
#     responses=ERROR_RESPONSES,
# )
# def predict(
#     customer: CustomerFeatures,
#     model: SegmentationModel = Depends(get_model),
# ) -> PredictionResponse:
#     """Assign one customer to a segment.

#     Body: five behavioural features. Out-of-range values are rejected with 422
#     before reaching the model.
#     """
#     try:
#         result = model.predict([customer.to_feature_vector()])[0]
#     except ValueError as exc:
#         raise HTTPException(status_code=400, detail=str(exc)) from exc
#     return PredictionResponse.model_validate(result)

# @app.post(
#     "/predict/batch",
#     tags=["prediction"],
#     summary="Segment up to 100 customers",
#     response_model=BatchPredictionResponse,
#     responses=ERROR_RESPONSES,
# )
# def predict_batch(
#     payload: BatchPredictRequest,
#     model: SegmentationModel = Depends(get_model),
# ) -> BatchPredictionResponse:
#     """Assign many customers in a single request.

#     One vectorised scaler + predict call, rather than a loop of single
#     predictions, so cost stays flat as the batch grows.
#     """
#     try:
#         rows: List[List[float]] = [c.to_feature_vector() for c in payload.customers]
#         results = model.predict(rows)
#     except ValueError as exc:
#         raise HTTPException(status_code=400, detail=str(exc)) from exc

#     return BatchPredictionResponse(
#         count=len(results),
#         predictions=[PredictionResponse.model_validate(r) for r in results],
#     )

# @app.get(
#     "/segments",
#     tags=["model"],
#     summary="List all segments",
#     response_model=SegmentsResponse,
#     responses=ERROR_RESPONSES,
# )
# def list_segments(model: SegmentationModel = Depends(get_model)) -> SegmentsResponse:
#     """List every segment with its size and centroid in original feature units."""
#     return SegmentsResponse(n_clusters=model.n_clusters, segments=model.segments())

# @app.get(
#     "/model/info",
#     tags=["model"],
#     summary="Model provenance and metrics",
#     response_model=ModelInfoResponse,
#     responses=ERROR_RESPONSES,
# )
# def model_info(model: SegmentationModel = Depends(get_model)) -> ModelInfoResponse:
#     """Describe the loaded model: type, features, training metrics and versions."""
#     return ModelInfoResponse.model_validate(model.info())

"""FastAPI application exposing the trained customer segmentation model.

Endpoints:
    GET  /              service index
    GET  /health        readiness + model metadata
    POST /predict       segment a single customer
    POST /predict/batch segment up to 100 customers in one call
    GET  /segments      all segments with centroids and sizes
    GET  /model/info    model provenance and training metrics

The model is loaded once during application startup (lifespan). If the
artifact is missing or malformed the process exits immediately with a clear
message instead of serving a broken service.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from app.model import ModelLoadError, SegmentationModel
from app.schemas import (
    BatchPredictionResponse,
    BatchPredictRequest,
    CustomerFeatures,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    SegmentsResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")

SERVICE_NAME = "customer-segmentation-api"
SERVICE_VERSION = "1.0.0"

# Optional override, e.g. MODEL_PATH=/models/other.pkl under Docker or in CI.
MODEL_PATH_ENV = "MODEL_PATH"

# Where to document an error we raise ourselves. FastAPI's built-in 422 body
# for malformed requests is left as-is (it is the standard shape).
ERROR_RESPONSES: Dict[int | str, Dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Feature values could not be used."},
    503: {"model": ErrorResponse, "description": "Model is not loaded."},
}

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup; fail fast if it cannot be loaded."""
    model_path = os.getenv(MODEL_PATH_ENV) or None
    logger.info(
        "Starting %s v%s (model path: %s)",
        SERVICE_NAME,
        SERVICE_VERSION,
        model_path or "default",
    )

    try:
        model = SegmentationModel.load(model_path)
    except ModelLoadError as exc:
        # Re-raised on purpose: crash now, not on the first user request.
        logger.critical("Startup aborted: %s", exc)
        raise

    app.state.model = model
    logger.info(
        "Ready: %s | %d clusters | features=%s",
        type(model.model).__name__,
        model.n_clusters,
        ",".join(model.feature_names),
    )
    try:
        yield
    finally:
        app.state.model = None
        logger.info("Shutdown complete.")

app = FastAPI(
    title="Customer Segmentation API",
    description=(
        "KMeans-based customer segmentation served over HTTP. "
        "Customers are described by five behavioural features and assigned to "
        "one of four segments. Trained offline by train_model.py; the fitted "
        "scaler and model are loaded from a single joblib artifact at startup."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
    contact={"name": "Customer Segmentation API"},
)

def get_model(request: Request) -> SegmentationModel:
    """Dependency that returns the loaded model, or 503 if unavailable."""
    model = getattr(request.app.state, "model", None)
    if model is None:  # pragma: no cover - only reachable if lifespan is bypassed
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return model

INDEX_PAGE = Path(__file__).parent / "index.html"


@app.get("/", tags=["meta"], summary="Service index")
def root(request: Request) -> Response:
    """Show the project page in a browser; return the JSON index to API clients."""
    if "text/html" in request.headers.get("accept", "") and INDEX_PAGE.is_file():
        return HTMLResponse(INDEX_PAGE.read_text(encoding="utf-8"))
    return JSONResponse(
        {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "status": "running",
            "endpoints": [
                "GET  /",
                "GET  /health",
                "POST /predict",
                "POST /predict/batch",
                "GET  /segments",
                "GET  /model/info",
                "GET  /docs",
                "GET  /openapi.json",
            ],
        }
    )

@app.get(
    "/health",
    tags=["meta"],
    summary="Readiness probe",
    response_model=HealthResponse,
    responses=ERROR_RESPONSES,
)
def health(model: SegmentationModel = Depends(get_model)) -> HealthResponse:
    """Report readiness plus which model artifact is loaded."""
    return HealthResponse.model_validate(model.health())

@app.post(
    "/predict",
    tags=["prediction"],
    summary="Segment a single customer",
    response_model=PredictionResponse,
    responses=ERROR_RESPONSES,
)
def predict(
    customer: CustomerFeatures,
    model: SegmentationModel = Depends(get_model),
) -> PredictionResponse:
    """Assign one customer to a segment.

    Body: five behavioural features. Out-of-range values are rejected with 422
    before reaching the model.
    """
    try:
        result = model.predict([customer.to_feature_vector()])[0]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionResponse.model_validate(result)

@app.post(
    "/predict/batch",
    tags=["prediction"],
    summary="Segment up to 100 customers",
    response_model=BatchPredictionResponse,
    responses=ERROR_RESPONSES,
)
def predict_batch(
    payload: BatchPredictRequest,
    model: SegmentationModel = Depends(get_model),
) -> BatchPredictionResponse:
    """Assign many customers in a single request.

    One vectorised scaler + predict call, rather than a loop of single
    predictions, so cost stays flat as the batch grows.
    """
    try:
        rows: List[List[float]] = [c.to_feature_vector() for c in payload.customers]
        results = model.predict(rows)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return BatchPredictionResponse(
        count=len(results),
        predictions=[PredictionResponse.model_validate(r) for r in results],
    )

@app.get(
    "/segments",
    tags=["model"],
    summary="List all segments",
    response_model=SegmentsResponse,
    responses=ERROR_RESPONSES,
)
def list_segments(model: SegmentationModel = Depends(get_model)) -> SegmentsResponse:
    """List every segment with its size and centroid in original feature units."""
    return SegmentsResponse(n_clusters=model.n_clusters, segments=model.segments())

@app.get(
    "/model/info",
    tags=["model"],
    summary="Model provenance and metrics",
    response_model=ModelInfoResponse,
    responses=ERROR_RESPONSES,
)
def model_info(model: SegmentationModel = Depends(get_model)) -> ModelInfoResponse:
    """Describe the loaded model: type, features, training metrics and versions."""
    return ModelInfoResponse.model_validate(model.info())