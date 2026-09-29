"""Train the customer segmentation model.

Generates a reproducible synthetic customer dataset, fits a KMeans model on
standardised features, and writes the scaler + model + metadata bundle to
model/customer_model.pkl.

Run from the project root:  python train_model.py
"""

# Pin BLAS/OpenMP to a single thread BEFORE numpy/sklearn are imported.
# KMeans accumulates squared distances in parallel chunks, and floating point
# addition is not associative, so the thread count can shift results in the
# last decimal place. Pinning makes this script produce the same pickle on
# Windows, in CI and inside Docker.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import platform  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
RANDOM_SEED = 42
N_CLUSTERS = 4
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "customer_model.pkl"

FEATURES = [
    "age",
    "annual_income_k",
    "spending_score",
    "purchase_frequency",
    "avg_order_value",
]

# Hard bounds applied after sampling, so the data looks like real data.
FEATURE_BOUNDS = {
    "age": (18, 75),
    "annual_income_k": (15.0, 200.0),
    "spending_score": (1, 100),
    "purchase_frequency": (1, 30),
    "avg_order_value": (10.0, 400.0),
}

# Four latent customer archetypes. The generator knows these; the clustering
# model is never told them, so we can score how well it recovered them.
ARCHETYPES = [
    {
        "archetype": "young_digital",
        "n": 300,
        "age": (26, 5),
        "annual_income_k": (32, 8),
        "spending_score": (72, 12),
        "purchase_frequency": (22, 5),
        "avg_order_value": (28, 8),
    },
    {
        "archetype": "mainstream_family",
        "n": 350,
        "age": (41, 7),
        "annual_income_k": (62, 12),
        "spending_score": (52, 12),
        "purchase_frequency": (13, 4),
        "avg_order_value": (75, 20),
    },
    {
        "archetype": "affluent_conservative",
        "n": 200,
        "age": (53, 8),
        "annual_income_k": (112, 20),
        "spending_score": (38, 11),
        "purchase_frequency": (6, 3),
        "avg_order_value": (205, 55),
    },
    {
        "archetype": "premium_loyalist",
        "n": 150,
        "age": (46, 6),
        "annual_income_k": (96, 15),
        "spending_score": (85, 9),
        "purchase_frequency": (18, 4),
        "avg_order_value": (150, 35),
    },
]


def generate_customers():
    """Build the synthetic dataset. Returns (X_raw, archetype_labels)."""
    rng = np.random.default_rng(RANDOM_SEED)
    blocks, truth = [], []

    for idx, spec in enumerate(ARCHETYPES):
        n = spec["n"]
        block = {f: rng.normal(spec[f][0], spec[f][1], n) for f in FEATURES}
        blocks.append(np.column_stack([block[f] for f in FEATURES]))
        truth.append(np.full(n, idx, dtype=int))

    X = np.vstack(blocks)
    y_truth = np.concatenate(truth)

    # Clip to plausible bounds.
    for j, f in enumerate(FEATURES):
        lo, hi = FEATURE_BOUNDS[f]
        X[:, j] = np.clip(X[:, j], lo, hi)

    # Round to the precision a real CRM export would have.
    X[:, FEATURES.index("age")] = np.round(X[:, FEATURES.index("age")])
    X[:, FEATURES.index("annual_income_k")] = np.round(
        X[:, FEATURES.index("annual_income_k")], 1
    )
    X[:, FEATURES.index("spending_score")] = np.round(
        X[:, FEATURES.index("spending_score")]
    )
    X[:, FEATURES.index("purchase_frequency")] = np.round(
        X[:, FEATURES.index("purchase_frequency")]
    )
    X[:, FEATURES.index("avg_order_value")] = np.round(
        X[:, FEATURES.index("avg_order_value")], 2
    )

    # Shuffle so records are not ordered by archetype.
    order = rng.permutation(len(X))
    return X[order], y_truth[order]


def evaluate_k(X_scaled, seed):
    """Inertia and silhouette for a range of k, for the elbow plot / README."""
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(X_scaled)
        rows.append((k, km.inertia_, silhouette_score(X_scaled, labels)))
    return rows


def main():
    print("=" * 68)
    print("Customer Segmentation - Model Training")
    print("=" * 68)
    print(f"python        : {platform.python_version()}")
    print(f"scikit-learn  : {sklearn.__version__}")
    print(f"numpy         : {np.__version__}")
    print(f"random seed   : {RANDOM_SEED}")
    print(f"clusters (k)  : {N_CLUSTERS}")
    print("-" * 68)

    X_raw, y_truth = generate_customers()
    print(f"Generated {X_raw.shape[0]} customers x {X_raw.shape[1]} features")

    scaler = StandardScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw)
    print("Scaled features (zero mean, unit variance)")

    print("-" * 68)
    print("Choosing k (inertia / silhouette):")
    for k, inertia, sil in evaluate_k(X_scaled, RANDOM_SEED):
        marker = "  <-- selected" if k == N_CLUSTERS else ""
        print(f"  k={k}  inertia={inertia:9.2f}  silhouette={sil:.4f}{marker}")

    model = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_SEED)
    raw_labels = model.fit_predict(X_scaled)

    # KMeans numbers its clusters arbitrarily. Relabel them in a stable order
    # (highest average income first) so segment IDs mean the same thing on
    # every machine and after every retrain.
    income_idx = FEATURES.index("annual_income_k")
    centroids_raw = scaler.inverse_transform(model.cluster_centers_)
    order = np.argsort(-centroids_raw[:, income_idx])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))

    model.cluster_centers_ = model.cluster_centers_[order]
    labels = inverse[raw_labels]
    model.labels_ = labels

    segment_names = {
        0: "Affluent Conservatives",
        1: "Premium Loyalists",
        2: "Mainstream Families",
        3: "Young Digital Spenders",
    }
    segment_descriptions = {
        0: "High income, low engagement. Older, high basket value but rarely shop.",
        1: "High income, high engagement, frequent buyers with premium baskets.",
        2: "Middle income, moderate engagement. The core mass-market cohort.",
        3: "Young, lower income, high engagement. Frequent small-basket buyers.",
    }

    inertia = float(model.inertia_)
    silhouette = float(silhouette_score(X_scaled, labels))
    ari = float(adjusted_rand_score(y_truth, labels))

    print("-" * 68)
    print(f"Trained KMeans: inertia={inertia:.2f}  silhouette={silhouette:.4f}")
    print(f"Adjusted Rand Index vs known archetypes: {ari:.4f}")
    print("-" * 68)
    print("Segment profile (mean feature values):")
    header = (
        f"{'id':>2}  {'segment':<24}{'n':>5}"
        f"{'age':>7}{'income_k':>10}{'spend':>7}{'freq':>7}{'aov':>8}"
    )
    print(header)
    for cid in range(N_CLUSTERS):
        mask = labels == cid
        means = X_raw[mask].mean(axis=0)
        print(
            f"{cid:>2}  {segment_names[cid]:<24}{int(mask.sum()):>5}"
            f"{means[0]:>7.1f}{means[1]:>10.1f}{means[2]:>7.1f}"
            f"{means[3]:>7.1f}{means[4]:>8.1f}"
        )
    print("-" * 68)

    feature_ranges = {
        f: {"min": float(X_raw[:, j].min()), "max": float(X_raw[:, j].max())}
        for j, f in enumerate(FEATURES)
    }

    bundle = {
        "scaler": scaler,
        "model": model,
        "n_clusters": N_CLUSTERS,
        "feature_names": FEATURES,
        "feature_ranges": feature_ranges,
        "segment_names": segment_names,
        "segment_descriptions": segment_descriptions,
        "metrics": {
            "inertia": inertia,
            "silhouette": silhouette,
            "adjusted_rand_index": ari,
        },
        "training": {
            "n_samples": int(X_raw.shape[0]),
            "random_seed": RANDOM_SEED,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)

    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"Wrote {MODEL_PATH.relative_to(BASE_DIR)}  ({size_kb:.1f} KB)")
    print("=" * 68)
    print("OK - model trained and saved.")


if __name__ == "__main__":
    main()