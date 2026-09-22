install:
	python -m pip install -r requirements.txt

validate:
	python -m src.data.validation

train:
	python -m src.models.train

test:
	pytest -q

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

app:
	streamlit run app/streamlit_app.py --server.port 8501

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

dvc-repro:
	dvc repro

docker-build:
	docker build -t edupredict:latest .

docker-run:
	docker run --rm -p 8000:8000 -p 8501:8501 edupredict:latest
