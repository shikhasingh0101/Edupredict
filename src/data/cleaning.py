"""Deterministic cleaning stage for EduPredict."""

from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/StudentPerformanceFactors.csv")
OUTPUT_PATH = Path("data/processed/student_performance_clean.csv")
TARGET = "Exam_Score"


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Preserve the raw dataset unchanged. The supplied dataset contains one
    # target value outside the expected 0–100 exam-score range (101).
    out = out[(out[TARGET] >= 0) & (out[TARGET] <= 100)].copy()

    # Do not manually impute predictors here. Imputation belongs inside the
    # training pipeline and is fitted only on training data.
    return out


def main():
    df = pd.read_csv(RAW_PATH)
    cleaned = clean_dataset(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_PATH, index=False)

    print(f"Raw rows: {len(df)}")
    print(f"Clean rows: {len(cleaned)}")
    print(f"Rows removed: {len(df) - len(cleaned)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
