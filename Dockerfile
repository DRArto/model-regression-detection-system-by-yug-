FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY prompts ./prompts
COPY data ./data
COPY reports ./reports

RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

CMD ["python", "-m", "model_regression.cli", "--prompt", "prompts/v1.yaml", "--dataset", "data/golden_dataset_v1.json"]
