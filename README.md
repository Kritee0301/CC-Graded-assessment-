# Customer Segmentation API

A KMeans model that assigns customers to one of four segments, served over a FastAPI REST API, packaged as a Docker image, and deployed to the cloud.

**Live UI:** <https://cc-graded-assessment.onrender.com/ui> · **API docs:** <https://cc-graded-assessment.onrender.com/docs> · **Health:** <https://cc-graded-assessment.onrender.com/health>

![Architecture](docs/architecture.png)

---

## What it is

- Trains on 1,000 synthetic customers with 5 behavioural features, clustering them into 4 segments (k = 4, chosen because silhouette peaks there).
- The trained model is a single file — `model/customer_model.pkl` (2,349 bytes) — holding the fitted scaler, the KMeans model and its metadata. It is committed to the repo and baked into the Docker image.
- The API loads that file once at startup and answers prediction requests over HTTP.
- A small browser UI is served by the API at `/ui` — enter a customer's features and it calls `/predict` and displays the assigned segment.
- Silhouette **0.4338** · Adjusted Rand Index **0.9238**.

Segments: Affluent Conservatives (198) · Premium Loyalists (150) · Mainstream Families (353) · Young Digital Spenders (299).

**Stack:** Python 3.11 · FastAPI · scikit-learn 1.9.1 · NumPy 2.4.6 · joblib · Pydantic · uvicorn · pytest · Docker · GitHub Actions · GHCR · Render

---

## Project structure

```
customer-segmentation-api/
├── app/main.py            # FastAPI routes and startup model loading
├── app/model.py           # loads the artifact, runs inference
├── app/schemas.py         # Pydantic request/response contracts
├── app/static/index.html  # browser UI served at /ui
├── model/customer_model.pkl
├── tests/test_api.py      # 52 tests
├── .github/workflows/ci-cd.yml
├── docs/architecture.png
├── train_model.py         # generates data, trains and saves the model
├── client_example.py      # HTTP client for the API
├── Dockerfile
└── requirements.txt
```

---

## Run locally

```bash
git clone https://github.com/kritee0301/customer-segmentation-api.git
cd customer-segmentation-api

python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000     # docs at http://localhost:8000/docs
```

Retrain the model with `python train_model.py`.

---

## Run with Docker

The image runs as a non-root user and has a healthcheck on `/health`.

```bash
docker build -t customer-segmentation-api .
docker run -d --name seg-api -p 8000:8000 customer-segmentation-api

# or pull the image CI publishes
docker pull ghcr.io/kritee0301/customer-segmentation-api:latest

# check it is healthy
docker inspect --format '{{.State.Health.Status}}' seg-api     # -> healthy
```

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Service index |
| GET | `/health` | Model loaded, cluster count, library version |
| POST | `/predict` | Segment one customer |
| POST | `/predict/batch` | Segment 1–100 customers at once |
| GET | `/segments` | Segments with sizes and centroids |
| GET | `/model/info` | Model type, features, metrics |

Features: `age` (18–75), `annual_income_k` (15–200), `spending_score` (1–100), `purchase_frequency` (1–30), `avg_order_value` (10–400). Anything outside these ranges, missing, or misspelled returns `422` before the model runs.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":40,"annual_income_k":61.0,"spending_score":51,"purchase_frequency":13,"avg_order_value":74.0}'
```

```json
{ "segment_id": 2, "segment_name": "Mainstream Families",
  "distance_to_centroid": 0.0623, "confidence_margin": 0.9731 }
```

`confidence_margin` is the gap to the second-closest segment (0 = on a boundary, 1 = unambiguous). It is not a probability.

The included client exercises every endpoint:

```bash
python client_example.py
python client_example.py https://cc-graded-assessment.onrender.com
```

A browser UI is also served at `/ui` (`http://localhost:8000/ui` locally, or `/ui` on the deployed URL).

---

## Tests

```bash
python -m pytest -v
```

**52 tests**, passing in ~1.7 s. They cover the endpoints, input validation (both edges of every range), batch behaviour, and model-loading failures — a missing, corrupt or reordered artifact.

---

## CI/CD

`.github/workflows/ci-cd.yml` runs on every push to `main` and on pull requests.

- **Test** — installs the pinned dependencies on Python 3.11, verifies the committed model artifact, runs the 52 tests.
- **Build and publish** — builds the Docker image, starts it, waits for Docker to report `healthy`, calls `/health` and `/predict` to confirm it really serves, then pushes to GHCR tagged with the commit SHA and `latest`. Pull requests build and smoke-test but do not publish.

---

## Deployment

Live on Render: <https://cc-graded-assessment.onrender.com>. Render builds this repository's Dockerfile, so the deployed container is the image CI validates.

> The free instance sleeps after 15 minutes of inactivity — the first request may take 30–60 seconds to wake.

The image is public and needs no authentication:

```bash
docker pull ghcr.io/kritee0301/customer-segmentation-api:latest
```

---

**Author:** Kritee — B.Tech, Computer Science Engineering · [@kritee0301](https://github.com/kritee0301)
**License:** MIT — see [LICENSE](LICENSE).
