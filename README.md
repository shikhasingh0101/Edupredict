# EduPredict

**AI Student Exam Performance Prediction & Academic Risk Intelligence System**

EduPredict is an end-to-end Feature Engineering and MLOps project that predicts `Exam_Score` from academic, behavioral, and contextual features.

## Dataset

Actual dataset:

`data/raw/StudentPerformanceFactors.csv`

The raw dataset is the single source of truth. No synthetic replacement data is used.

## Architecture

Raw Dataset → Validation → Feature Engineering → Preprocessing → Feature Selection → Optional PCA → Model → MLflow → Model Artifact → FastAPI → Streamlit

Supporting MLOps:

DVC → reproducibility  
GitHub Actions → CI/CD  
Docker → containerization  
Monitoring → operational metrics  
Drift Detection → retraining signal

## Local setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Validate

```bash
python -m src.data.validation
```

## Train

```bash
python -m src.models.train
```

## Test

```bash
pytest -q
```

## FastAPI

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API documentation:

`http://localhost:8000/docs`

## Streamlit

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

## MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## DVC

```bash
dvc repro
```

## Docker

```bash
docker build -t YOUR_DOCKERHUB_USERNAME/edupredict:latest .
docker run --rm -p 8000:8000 -p 8501:8501 YOUR_DOCKERHUB_USERNAME/edupredict:latest
```

Do not replace the Docker Hub username with a fabricated value.

## Important

Model metrics, PCA findings, MLflow runs, Docker Hub publishing, and deployment status must only be documented after actual execution.

Predictions are project decision-support outputs and are not validated educational, psychological, or medical diagnoses.
