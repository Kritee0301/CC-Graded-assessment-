"""Test suite for the customer segmentation API.

Covers the happy paths, the validation boundary conditions, the batch
contract, and the model-loader failure modes (missing / corrupt / mismatched
artifacts) that would otherwise only show up in production.

Run from the project root:  python -m pytest -v
"""

from __future__ import annotations

from typing import Any, Dict

import joblib
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import DEFAULT_MODEL_PATH, ModelLoadError, SegmentationModel
from app.schemas import FEATURE_ORDER

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client() -> TestClient:
    """A TestClient with the real app, lifespan included (model loaded once)."""
    with TestClient(app) as test_client:
        yield test_client

# Canonical sample customers, reused across tests.
MIDSTREAM = {
    "age": 40,
    "annual_income_k": 61.0,
    "spending_score": 51,
    "purchase_frequency": 13,
    "avg_order_value": 74.0,
}
YOUNG_DIGITAL = {
    "age": 27,
    "annual_income_k": 33.0,
    "spending_score": 74,
    "purchase_frequency": 23,
    "avg_order_value": 30.0,
}
AFFLUENT = {
    "age": 54,
    "annual_income_k": 116.0,
    "spending_score": 37,
    "purchase_frequency": 6,
    "avg_order_value": 210.0,
}
PREMIUM_LOYALIST = {
    "age": 46,
    "annual_income_k": 98.0,
    "spending_score": 86,
    "purchase_frequency": 18,
    "avg_order_value": 152.0,
}

# --------------------------------------------------------------------------
# Meta endpoints
# --------------------------------------------------------------------------

def test_root_lists_endpoints(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert body["service"] == "customer-segmentation-api"
    assert any("/predict" in endpoint for endpoint in body["endpoints"])

def test_health_reports_loaded_model(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_clusters"] == 4
    assert body["features"] == list(FEATURE_ORDER)
    # Health must not leak absolute server paths.
    assert not body["model_path"].startswith("/")
    assert ":" not in body["model_path"]

def test_openapi_schema_exposes_expected_paths(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    for path in ("/", "/health", "/predict", "/predict/batch", "/segments", "/model/info"):
        assert path in paths, f"{path} missing from OpenAPI schema"
    assert schema["info"]["version"] == "1.0.0"

# --------------------------------------------------------------------------
# Single prediction
# --------------------------------------------------------------------------

def test_predict_returns_expected_segment(client: TestClient) -> None:
    response = client.post("/predict", json=MIDSTREAM)
    assert response.status_code == 200
    body = response.json()

    assert body["segment_id"] == 2
    assert body["segment_name"] == "Mainstream Families"
    assert body["segment_description"]
    assert body["distance_to_centroid"] == pytest.approx(0.0623, abs=1e-3)

def test_predict_assigns_distinct_segments_to_distinct_customers(
    client: TestClient,
) -> None:
    expected = {
        3: YOUNG_DIGITAL,
        2: MIDSTREAM,
        0: AFFLUENT,
        1: PREMIUM_LOYALIST,
    }
    for expected_id, customer in expected.items():
        body = client.post("/predict", json=customer).json()
        assert body["segment_id"] == expected_id, (
            f"{customer} was assigned {body['segment_id']} "
            f"({body['segment_name']}), expected {expected_id}"
        )

def test_predict_response_invariants(client: TestClient) -> None:
    body = client.post("/predict", json=MIDSTREAM).json()
    distances: Dict[str, float] = body["segment_distances"]

    assert len(distances) == 4
    assert all(value >= 0 for value in distances.values())
    # The assigned segment must be the nearest one.
    assert distances[body["segment_name"]] == pytest.approx(
        body["distance_to_centroid"], abs=1e-6
    )
    assert distances[body["segment_name"]] == min(distances.values())
    # Margin is a ratio, not a probability.
    assert 0.0 <= body["confidence_margin"] <= 1.0

def test_predict_is_deterministic(client: TestClient) -> None:
    first = client.post("/predict", json=MIDSTREAM).json()
    second = client.post("/predict", json=MIDSTREAM).json()
    assert first == second

@pytest.mark.parametrize(
    "field,value",
    [
        ("age", 17),
        ("age", 76),
        ("annual_income_k", 14.9),
        ("annual_income_k", 200.1),
        ("spending_score", 0),
        ("spending_score", 101),
        ("purchase_frequency", 0),
        ("purchase_frequency", 31),
        ("avg_order_value", 9.9),
        ("avg_order_value", 400.1),
    ],
)
def test_predict_rejects_out_of_range_values(
    client: TestClient, field: str, value: float
) -> None:
    payload = dict(MIDSTREAM)
    payload[field] = value
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert any(
        error["loc"][-1] == field for error in response.json()["detail"]
    ), f"422 did not point at {field}: {response.json()}"

@pytest.mark.parametrize(
    "field,value",
    [
        ("age", 18),
        ("age", 75),
        ("annual_income_k", 15.0),
        ("annual_income_k", 200.0),
        ("spending_score", 1),
        ("spending_score", 100),
        ("purchase_frequency", 1),
        ("purchase_frequency", 30),
        ("avg_order_value", 10.0),
        ("avg_order_value", 400.0),
    ],
)
def test_predict_accepts_exact_boundary_values(
    client: TestClient, field: str, value: float
) -> None:
    """The documented bounds are inclusive; both edges must be accepted."""
    payload = dict(MIDSTREAM)
    payload[field] = value
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    assert 0 <= response.json()["segment_id"] <= 3

def test_predict_rejects_missing_field(client: TestClient) -> None:
    payload = dict(MIDSTREAM)
    del payload["spending_score"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert any(
        error["type"] == "missing" for error in response.json()["detail"]
    )

def test_predict_rejects_unknown_field(client: TestClient) -> None:
    """extra='forbid' means a typo'd field is an error, not silently ignored."""
    payload = dict(MIDSTREAM)
    payload["income"] = 61.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert any(
        error["type"] == "extra_forbidden" for error in response.json()["detail"]
    )

def test_predict_rejects_non_numeric_value(client: TestClient) -> None:
    payload = dict(MIDSTREAM)
    payload["age"] = "forty"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_rejects_empty_body(client: TestClient) -> None:
    response = client.post("/predict", json={})
    assert response.status_code == 422

def test_predict_returns_400_when_model_rejects_vector(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ValueError from the model layer must surface as 400, not a 500."""
    model = app.state.model
    assert model is not None

    def explode(rows: Any) -> Any:
        raise ValueError("simulated model failure")

    monkeypatch.setattr(model, "predict", explode)
    response = client.post("/predict", json=MIDSTREAM)
    assert response.status_code == 400
    assert "simulated model failure" in response.json()["detail"]

# --------------------------------------------------------------------------
# Batch prediction
# --------------------------------------------------------------------------

def test_batch_predicts_all_customers_in_order(client: TestClient) -> None:
    payload = {
        "customers": [YOUNG_DIGITAL, MIDSTREAM, AFFLUENT, PREMIUM_LOYALIST]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()

    assert body["count"] == 4
    assert len(body["predictions"]) == 4
    assert [p["segment_id"] for p in body["predictions"]] == [3, 2, 0, 1]

def test_batch_matches_single_prediction(client: TestClient) -> None:
    """Batch results must agree with one-at-a-time predictions."""
    for customer in (YOUNG_DIGITAL, MIDSTREAM, AFFLUENT, PREMIUM_LOYALIST):
        single = client.post("/predict", json=customer).json()

        batch = client.post("/predict/batch", json={"customers": [customer]}).json()
        assert batch["count"] == 1
        assert batch["predictions"][0] == single

def test_batch_accepts_maximum_size(client: TestClient) -> None:
    payload = {"customers": [MIDSTREAM] * 100}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    assert response.json()["count"] == 100

def test_batch_rejects_empty_list(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code == 422

def test_batch_rejects_oversized_list(client: TestClient) -> None:
    payload = {"customers": [MIDSTREAM] * 101}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422

def test_batch_rejects_invalid_customer_in_list(client: TestClient) -> None:
    """One bad record invalidates the request rather than being skipped."""
    bad = dict(MIDSTREAM)
    bad["age"] = 99
    payload = {"customers": [MIDSTREAM, bad]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422

def test_batch_duplicate_records_get_identical_predictions(
    client: TestClient,
) -> None:
    """Repeated records must produce identical results.

    A common batch-prediction bug is per-item state leaking across a batch,
    which makes the same record score differently depending on its position.
    """
    payload = {"customers": [MIDSTREAM] * 5}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert all(p["segment_id"] == predictions[0]["segment_id"] for p in predictions)
    assert all(
        p["distance_to_centroid"] == predictions[0]["distance_to_centroid"]
        for p in predictions
    )

# --------------------------------------------------------------------------
# Model introspection endpoints
# --------------------------------------------------------------------------

def test_segments_endpoint_structure(client: TestClient) -> None:
    response = client.get("/segments")
    assert response.status_code == 200
    body = response.json()

    assert body["n_clusters"] == 4
    assert len(body["segments"]) == 4
    assert [s["segment_id"] for s in body["segments"]] == [0, 1, 2, 3]
    assert sum(s["customer_count"] for s in body["segments"]) == 1000

    for segment in body["segments"]:
        assert segment["name"]
        assert segment["description"]
        assert set(segment["centroid"]) == set(FEATURE_ORDER)
        assert all(value >= 0 for value in segment["centroid"].values())

def test_segments_are_ordered_by_descending_income(client: TestClient) -> None:
    """Segment ids are stable: 0 is always the highest-income segment."""
    segments = client.get("/segments").json()["segments"]
    incomes = [segment["centroid"]["annual_income_k"] for segment in segments]
    assert incomes == sorted(incomes, reverse=True)

def test_segment_names_are_stable(client: TestClient) -> None:
    segments = client.get("/segments").json()["segments"]
    assert [segment["name"] for segment in segments] == [
        "Affluent Conservatives",
        "Premium Loyalists",
        "Mainstream Families",
        "Young Digital Spenders",
    ]

def test_model_info_reports_training_metrics(client: TestClient) -> None:
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()

    assert body["model_type"] == "KMeans"
    assert body["n_clusters"] == 4
    assert body["features"] == list(FEATURE_ORDER)

    metrics = body["metrics"]
    assert metrics["inertia"] > 0
    assert 0.0 < metrics["silhouette"] < 1.0
    # The model should recover the generator's archetypes convincingly.
    assert metrics["adjusted_rand_index"] > 0.8

    training = body["training"]
    assert training["n_samples"] == 1000
    assert training["random_seed"] == 42
    assert training["sklearn_version"]

# --------------------------------------------------------------------------
# Routing behaviour
# --------------------------------------------------------------------------

def test_unknown_path_returns_404(client: TestClient) -> None:
    assert client.get("/nope").status_code == 404

def test_wrong_method_returns_405(client: TestClient) -> None:
    assert client.get("/predict").status_code == 405

# --------------------------------------------------------------------------
# Model loader failure modes (unit level, no HTTP)
# --------------------------------------------------------------------------

def test_load_from_missing_path_raises(tmp_path: Any) -> None:
    with pytest.raises(ModelLoadError, match="not found"):
        SegmentationModel.load(tmp_path / "absent.pkl")

def test_load_rejects_corrupt_file(tmp_path: Any) -> None:
    bogus = tmp_path / "corrupt.pkl"
    bogus.write_bytes(b"this is definitely not a joblib pickle")
    with pytest.raises(ModelLoadError, match="Could not read"):
        SegmentationModel.load(bogus)

def test_load_rejects_non_dict_pickle(tmp_path: Any) -> None:
    path = tmp_path / "list.pkl"
    joblib.dump(["not", "a", "bundle"], path)
    with pytest.raises(ModelLoadError, match="dictionary bundle"):
        SegmentationModel.load(path)

def test_load_rejects_bundle_with_missing_keys(tmp_path: Any) -> None:
    bundle: Dict[str, Any] = dict(joblib.load(DEFAULT_MODEL_PATH))
    del bundle["scaler"]
    path = tmp_path / "incomplete.pkl"
    joblib.dump(bundle, path)
    with pytest.raises(ModelLoadError, match="missing required keys"):
        SegmentationModel.load(path)

def test_load_rejects_feature_order_mismatch(tmp_path: Any) -> None:
    """If bundle order != FEATURE_ORDER the scaler would mis-scale silently."""
    bundle: Dict[str, Any] = dict(joblib.load(DEFAULT_MODEL_PATH))
    bundle["feature_names"] = list(reversed(list(FEATURE_ORDER)))
    path = tmp_path / "reordered.pkl"
    joblib.dump(bundle, path)
    with pytest.raises(ModelLoadError, match="Feature order mismatch"):
        SegmentationModel.load(path)

def test_valid_bundle_loads_and_predicts(tmp_path: Any) -> None:
    """Sanity check that the loader accepts the real artifact."""
    model = SegmentationModel.load(DEFAULT_MODEL_PATH)
    assert model.n_clusters == 4
    assert model.feature_names == list(FEATURE_ORDER)

    result = model.predict([list(MIDSTREAM.values())])[0]
    assert result["segment_name"] == "Mainstream Families"

def test_startup_fails_when_model_path_is_broken(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing artifact must abort startup, not serve a broken service."""
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "does-not-exist.pkl"))
    with pytest.raises(ModelLoadError):
        with TestClient(app):
            pass