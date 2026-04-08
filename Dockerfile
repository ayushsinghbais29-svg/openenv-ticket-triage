FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py .
COPY openenv.yaml .

# Default environment variables (override at runtime)
ENV API_BASE_URL=https://api.openai.com/v1
ENV MODEL_NAME=gpt-3.5-turbo
ENV HF_TOKEN=""

CMD ["python", "inference.py"]
