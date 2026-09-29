"""Model loading and inference for the customer segmentation API.

train_model.py writes a single joblib bundle containing the fitted scaler,
the fitted KMeans model and its metadata. This module loads that bundle once
at application startup and exposes narrow, validated inference helpers.

There is no database and no network access here: everything is in-memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import sklearn
from numpy.typing import NDArray

from app.schemas import FEATURE_ORDER

logger = logging.getLogger(__name__)

# Project root: this file lives at <root>/app/model.py
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "model" / "customer_model.pkl"

# Keys the bundle must contain for this service to function at all.
REQUIRED_BUNDLE_KEYS = (
    "scaler",
    "model",
    "n_clusters",
    "feature_names",
    "segment_names",
)

class ModelLoadError(RuntimeError):
    """Raised when the model artifact is missing, unreadable or malformed.

    Startup fails loudly on this error instead of serving a broken service.
    """

class SegmentationModel:
    """Validated wrapper around the training bundle."""

    def __init__(self, bundle: Dict[str, Any], path: Path) -> None:
        self._bundle = bundle
        self.path = path

        self.scaler = bundle["scaler"]
        self.model = bundle["model"]
        self.n_clusters: int = int(bundle["n_clusters"])
        self.feature_names: List[str] = [str(f) for f in bundle["feature_names"]]
        self.segment_names: Dict[int, str] = {
            int(k): str(v) for k, v in bundle["segment_names"].items()
        }
        self.segment_descriptions: Dict[int, str] = {
            int(k): str(v)
            for k, v in (bundle.get("segment_descriptions") or {}).items()
        }
        self.metrics: Dict[str, float] = dict(bundle.get("metrics") or {})
        self.training: Dict[str, Any] = dict(bundle.get("training") or {})
        self.feature_ranges: Dict[str, Dict[str, float]] = dict(
            bundle.get("feature_ranges") or {}
        )

        # Centroids converted back to original feature units, for reporting.
        self.centroids_raw: NDArray[np.float64] = self.scaler.inverse_transform(
            self.model.cluster_centers_
        )

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> "SegmentationModel":
        """Load and validate the bundle, or raise ModelLoadError."""
        model_path = Path(path) if path is not None else DEFAULT_MODEL_PATH

        if not model_path.exists():
            raise ModelLoadError(
                f"Model artifact not found at '{model_path}'. "
                "Run 'python train_model.py' from the project root to generate it, "
                "and make sure model/customer_model.pkl is committed to git "
                "(it is required by both the Docker build and CI)."
            )

        try:
            bundle = joblib.load(model_path)
        except Exception as exc:  # corrupt file, version mismatch, permissions...
            raise ModelLoadError(
                f"Could not read the model artifact '{model_path}': "
                f"{type(exc).__name__}: {exc}. Re-run 'python train_model.py'."
            ) from exc

        if not isinstance(bundle, dict):
            raise ModelLoadError(
                f"'{model_path}' did not contain a dictionary bundle "
                f"(got {type(bundle).__name__}). Re-run 'python train_model.py'."
            )

        missing = [key for key in REQUIRED_BUNDLE_KEYS if key not in bundle]
        if missing:
            raise ModelLoadError(
                f"Model bundle '{model_path}' is missing required keys: {missing}. "
                "Re-run 'python train_model.py'."
            )

        saved_version = str((bundle.get("training") or {}).get("sklearn_version", ""))
        if saved_version and saved_version.split(".")[:2] != sklearn.__version__.split(".")[:2]:
            logger.warning(
                "Model was trained with scikit-learn %s but this environment has %s. "
                "Predictions may differ. Re-run 'python train_model.py' with the "
                "versions pinned in requirements.txt.",
                saved_version,
                sklearn.__version__,
            )

        instance = cls(bundle, model_path)

        if instance.feature_names != list(FEATURE_ORDER):
            raise ModelLoadError(
                "Feature order mismatch: the model bundle expects "
                f"{instance.feature_names} but app/schemas.py declares "
                f"{list(FEATURE_ORDER)}. The scaler would standardise the wrong "
                "columns. Re-run 'python train_model.py' or fix FEATURE_ORDER."
            )

        instance._smoke_test()
        logger.info(
            "Loaded %s model: %d clusters, %d features, trained with scikit-learn %s",
            type(instance.model).__name__,
            instance.n_clusters,
            instance.n_features,
            instance.sklearn_version,
        )
        return instance

    def _smoke_test(self) -> None:
        """Prove the scaler and model can actually run, at startup, not at request time."""
        probe = np.zeros((1, self.n_features), dtype=float)
        try:
            self.scaler.transform(probe)
            self.model.predict(self.scaler.transform(probe))
        except Exception as exc:
            raise ModelLoadError(
                "The loaded model could not run a smoke-test prediction: "
                f"{type(exc).__name__}: {exc}. Re-run 'python train_model.py'."
            ) from exc

    # ------------------------------------------------------------ properties
    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def sklearn_version(self) -> str:
        return str(self.training.get("sklearn_version", "unknown"))

    # ------------------------------------------------------------- inference
    def predict(self, rows: List[List[float]]) -> List[Dict[str, Any]]:
        """Predict a segment for each row of raw (unscaled) feature values.

        Returns one dict per row, ready to be validated by PredictionResponse.
        """
        matrix = self._as_matrix(rows)
        scaled = self.scaler.transform(matrix)
        labels = self.model.predict(scaled)

        # Distance from every row to every centroid, in standardised space.
        deltas = scaled[:, np.newaxis, :] - self.model.cluster_centers_[np.newaxis, :, :]
        distances = np.linalg.norm(deltas, axis=2)

        results: List[Dict[str, Any]] = []
        for row_index, label in enumerate(labels):
            assigned = int(label)
            row_distances = distances[row_index]
            distance_to_assigned = float(row_distances[assigned])

            others = np.delete(row_distances, assigned)
            closest_other = float(others.min()) if others.size else float("inf")
            # Relative margin: how much further the runner-up segment is.
            margin = (
                0.0
                if closest_other <= 0.0
                else max(0.0, (closest_other - distance_to_assigned) / closest_other)
            )

            results.append(
                {
                    "segment_id": assigned,
                    "segment_name": self._segment_name(assigned),
                    "segment_description": self.segment_descriptions.get(assigned, ""),
                    "distance_to_centroid": round(distance_to_assigned, 4),
                    "confidence_margin": round(min(1.0, margin), 4),
                    "segment_distances": {
                        self._segment_name(cid): round(float(row_distances[cid]), 4)
                        for cid in range(self.n_clusters)
                    },
                }
            )
        return results

    def _as_matrix(self, rows: List[List[float]]) -> NDArray[np.float64]:
        """Convert raw input to a validated (n_samples, n_features) array."""
        if not rows:
            raise ValueError("At least one customer is required.")

        matrix = np.asarray(rows, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features per customer "
                f"({self.feature_names}), got an array of shape {matrix.shape}."
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Feature values must be finite numbers (no NaN or infinity).")
        return matrix

    def _segment_name(self, segment_id: int) -> str:
        return self.segment_names.get(segment_id, f"Segment {segment_id}")

    # -------------------------------------------------------------- reporting
    def segments(self) -> List[Dict[str, Any]]:
        """Every segment with its size and centroid, in original units."""
        labels = getattr(self.model, "labels_", None)
        if labels is not None and len(labels):
            sizes = np.bincount(np.asarray(labels), minlength=self.n_clusters)
        else:
            sizes = np.zeros(self.n_clusters, dtype=int)

        return [
            {
                "segment_id": cid,
                "name": self._segment_name(cid),
                "description": self.segment_descriptions.get(cid, ""),
                "customer_count": int(sizes[cid]) if cid < len(sizes) else 0,
                "centroid": {
                    name: round(float(self.centroids_raw[cid, index]), 2)
                    for index, name in enumerate(self.feature_names)
                },
            }
            for cid in range(self.n_clusters)
        ]

    def info(self) -> Dict[str, Any]:
        """Model provenance and performance, for the /model/info endpoint."""
        return {
            "model_type": type(self.model).__name__,
            "n_clusters": self.n_clusters,
            "features": self.feature_names,
            "feature_bounds": self.feature_ranges,
            "metrics": self.metrics,
            "training": self.training,
            "segments": self.segments(),
        }

    def health(self) -> Dict[str, Any]:
        """Readiness payload. Reports a relative path, never an absolute one."""
        try:
            display_path = str(self.path.relative_to(BASE_DIR))
        except ValueError:
            display_path = self.path.name

        return {
            "status": "ok",
            "model_loaded": True,
            "model_path": display_path,
            "sklearn_version": self.sklearn_version,
            "n_clusters": self.n_clusters,
            "features": self.feature_names,
        }