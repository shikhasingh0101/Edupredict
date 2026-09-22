from pathlib import Path
import pytest


def test_model_artifact_exists_after_training():
    if not Path("models/model.joblib").exists():
        pytest.skip("Train the model before running the full model test.")
    assert Path("models/model.joblib").stat().st_size > 0
