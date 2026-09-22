import pandas as pd
from src.data.validation import validate_dataset


def test_validation():
    df = pd.DataFrame({"x": [1, 2], "Exam_Score": [60, 70]})
    result = validate_dataset(df)
    assert result["rows"] == 2
    assert result["target"] == "Exam_Score"
