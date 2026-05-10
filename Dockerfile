FROM python:3.11-slim

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY training/ ./training/
COPY evaluation/ ./evaluation/
COPY params.yaml .
COPY dvc.yaml .

CMD ["python", "-m", "training.train"]