"""Dataset validation stage for EduPredict."""

from pathlib import Path
import json
import pandas as pd

EXPECTED_TARGET = "Exam_Score"
REPORT_PATH = Path("outputs/reports/data_validation.json")


def load_dataset(path="data/raw/StudentPerformanceFactors.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df


def validate_dataset(df: pd.DataFrame, target: str = EXPECTED_TARGET) -> dict:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    if df[target].isna().any():
        raise ValueError("Target contains missing values.")

    invalid_target_rows = int(((df[target] < 0) | (df[target] > 100)).sum())

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "rows_with_missing_values": int(df.isna().any(axis=1).sum()),
        "invalid_target_rows_outside_0_100": invalid_target_rows,
        "target": target,
        "numerical_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude="number").columns.tolist(),
    }


def main():
    df = load_dataset()
    report = validate_dataset(df)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
