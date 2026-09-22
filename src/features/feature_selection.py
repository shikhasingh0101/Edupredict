"""Leakage-safe feature selection stage for EduPredict.

The selection stage creates a reproducible feature list using only the
training portion of the development data. The final test set is never used.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

INPUT_PATH = Path("data/processed/student_performance_engineered.csv")
SELECTED_PATH = Path("outputs/metrics/selected_features.json")
IMPORTANCE_PATH = Path("outputs/metrics/permutation_feature_importance.csv")
TARGET = "Exam_Score"
RANDOM_STATE = 42


def main():
    df = pd.read_csv(INPUT_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.20, random_state=RANDOM_STATE
    )

    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = X_train.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)

    importance = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="neg_root_mean_squared_error",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)

    # Keep features with positive mean permutation importance. If none are
    # positive, retain all features rather than producing an empty model.
    selected = importance_df.loc[
        importance_df["importance_mean"] > 0, "feature"
    ].tolist()

    if not selected:
        selected = X.columns.tolist()

    SELECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)
    SELECTED_PATH.write_text(json.dumps({
        "selected_features": selected,
        "selection_rule": "positive_mean_permutation_importance",
        "random_state": RANDOM_STATE,
    }, indent=2))

    print(f"Selected {len(selected)} of {len(X.columns)} features.")
    print(selected)


if __name__ == "__main__":
    main()
