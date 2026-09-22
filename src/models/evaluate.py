"""Final test evaluation stage for EduPredict."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/processed/student_performance_engineered.csv")
MODEL_PATH = Path("models/edupredict_final_pipeline.joblib")
OUTPUT_PATH = Path("outputs/metrics/final_model_evaluation.json")
PREDICTIONS_PATH = Path("outputs/predictions/test_predictions.csv")
TARGET = "Exam_Score"


def main():
    df = pd.read_csv(DATA_PATH)

    with open("outputs/metrics/selected_features.json") as f:
        selection = json.load(f)

    features = [
        c for c in selection["selected_features"]
        if c in df.columns and c != TARGET
    ]

    X = df[features]
    y = df[TARGET]

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = joblib.load(MODEL_PATH)
    pred = model.predict(X_test)

    metrics = {
        "model": model.named_steps["model"].__class__.__name__,
        "test_rows": int(len(y_test)),
        "test_rmse": float(mean_squared_error(y_test, pred) ** 0.5),
        "test_mae": float(mean_absolute_error(y_test, pred)),
        "test_r2": float(r2_score(y_test, pred)),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({
        "actual_score": y_test.to_numpy(),
        "predicted_score": pred,
        "residual": y_test.to_numpy() - pred,
    }).to_csv(PREDICTIONS_PATH, index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
