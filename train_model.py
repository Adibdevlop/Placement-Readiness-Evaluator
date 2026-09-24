"""Reproduce the bundled CareerLens model and transparent synthetic dataset."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
FEATURE_NAMES = [
    "cgpa", "dsa_questions", "coding_skill", "aptitude_score",
    "communication_skill", "projects", "internships", "mock_interviews",
    "resume_score", "weekly_prep_hours",
]


def build_dataset(rows=2500, seed=42):
    rng = np.random.default_rng(seed)
    data = np.column_stack([
        rng.uniform(5, 10, rows),
        np.clip(rng.gamma(2.2, 100, rows), 0, 1000),
        rng.integers(1, 11, rows),
        rng.uniform(30, 100, rows),
        rng.integers(1, 11, rows),
        np.clip(rng.poisson(2.4, rows), 0, 15),
        np.clip(rng.poisson(0.8, rows), 0, 8),
        np.clip(rng.poisson(4.5, rows), 0, 50),
        rng.uniform(35, 100, rows),
        np.clip(rng.gamma(3.0, 5.5, rows), 0, 80),
    ])

    score = (
        data[:, 0] / 10 * 10
        + np.minimum(data[:, 1] / 400, 1) * 16
        + data[:, 2] / 10 * 16
        + data[:, 3] / 100 * 9
        + data[:, 4] / 10 * 12
        + np.minimum(data[:, 5] / 4, 1) * 10
        + np.minimum(data[:, 6] / 2, 1) * 7
        + np.minimum(data[:, 7] / 10, 1) * 7
        + data[:, 8] / 100 * 7
        + np.minimum(data[:, 9] / 25, 1) * 6
        + rng.normal(0, 2.5, rows)
    )
    return data, np.clip(score, 0, 100)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    x, y = build_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=220, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    header = ",".join(FEATURE_NAMES + ["readiness_score"])
    np.savetxt(DATA_DIR / "career_readiness_synthetic.csv", np.column_stack([x, y]), delimiter=",", header=header, comments="", fmt="%.3f")
    joblib.dump({
        "model": model,
        "feature_names": FEATURE_NAMES,
        "metrics": {
            "mae": round(float(mean_absolute_error(y_test, predictions)), 3),
            "r2": round(float(r2_score(y_test, predictions)), 3),
        },
        "dataset_note": "Synthetic educational dataset generated deterministically by train_model.py",
    }, MODEL_DIR / "careerlens_model.pkl", compress=3)
    print("Model saved", MODEL_DIR / "careerlens_model.pkl")


if __name__ == "__main__":
    main()

