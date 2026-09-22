import pandas as pd
from src.features.feature_engineering import add_features


def test_feature_engineering():
    df = pd.DataFrame({
        "Hours_Studied": [10],
        "Attendance": [80],
        "Sleep_Hours": [7],
    })
    out = add_features(df)
    assert out.loc[0, "Study_Attendance_Interaction"] == 800
    assert out.loc[0, "Study_Sleep_Interaction"] == 70
