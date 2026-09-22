"""Reproducible DVC training stage for EduPredict.

Model selection uses only train/validation data. The final test split is
created but never used to choose the model. The selected pipeline is then
refit on the complete development set and saved as the production artifact.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/processed/student_performance_engineered.csv")
SELECTION_PATH = Path("outputs/metrics/selected_features.json")
SUMMARY_PATH = Path("outputs/metrics/model_experiment_summary.json")
MODEL_PATH = Path("models/edupredict_final_pipeline.joblib")
TARGET = "Exam_Score"
RANDOM_STATE = 42


def make_preprocessor(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )),
        ]), categorical),
    ])


def make_model(name):
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        ),
    }
    return models[name]


def build_pipeline(X, name):
    return Pipeline([
        ("preprocessor", make_preprocessor(X)),
        ("model", make_model(name)),
    ])


def main():
    df = pd.read_csv(DATA_PATH)

    with open(SELECTION_PATH) as f:
        selection = json.load(f)

    selected = [
        c for c in selection["selected_features"]
        if c in df.columns and c != TARGET
    ]

    if not selected:
        selected = [c for c in df.columns if c != TARGET]

    X = df[selected].copy()
    y = df[TARGET].copy()

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.20, random_state=RANDOM_STATE
    )

    results = []

    for name in [
        "LinearRegression",
        "Ridge",
        "RandomForest",
        "ExtraTrees",
        "GradientBoosting",
        "HistGradientBoosting",
    ]:
        pipeline = build_pipeline(X_train, name)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_val)

        results.append({
            "model": name,
            "validation_rmse": mean_squared_error(y_val, pred) ** 0.5,
            "validation_mae": mean_absolute_error(y_val, pred),
            "validation_r2": r2_score(y_val, pred),
        })

    results_df = pd.DataFrame(results).sort_values(
        "validation_rmse"
    ).reset_index(drop=True)

    champion_name = results_df.iloc[0]["model"]

    # Refit only after model selection, using development data (train+val).
    final_pipeline = build_pipeline(X_dev, champion_name)
    final_pipeline.fit(X_dev, y_dev)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)

    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    results_df.to_csv(
        "outputs/metrics/model_experiment_results.csv",
        index=False,
    )

    SUMMARY_PATH.write_text(json.dumps({
        "champion_model": champion_name,
        "champion_validation_rmse": float(results_df.iloc[0]["validation_rmse"]),
        "champion_validation_mae": float(results_df.iloc[0]["validation_mae"]),
        "champion_validation_r2": float(results_df.iloc[0]["validation_r2"]),
        "selected_features": selected,
        "test_rows": int(len(y_test)),
        "note": "Test set was not used for model selection.",
    }, indent=2))

    print(results_df.to_string(index=False))
    print(f"\nChampion: {champion_name}")
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
