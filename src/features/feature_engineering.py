"""Feature-engineering stage for EduPredict."""

from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/student_performance_clean.csv")
OUTPUT_PATH = Path("data/processed/student_performance_engineered.csv")

ENGINEERED_FEATURES = [
    "Study_Attendance_Interaction",
    "Study_Sleep_Interaction",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Study_Attendance_Interaction"] = (
        out["Hours_Studied"] * out["Attendance"]
    )
    out["Study_Sleep_Interaction"] = (
        out["Hours_Studied"] * out["Sleep_Hours"]
    )
    return out


def feature_names(df: pd.DataFrame) -> list[str]:
    return [c for c in ENGINEERED_FEATURES if c in df.columns]


def main():
    df = pd.read_csv(INPUT_PATH)
    engineered = add_features(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_csv(OUTPUT_PATH, index=False)

    print(f"Input shape: {df.shape}")
    print(f"Output shape: {engineered.shape}")
    print(f"Added features: {ENGINEERED_FEATURES}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
