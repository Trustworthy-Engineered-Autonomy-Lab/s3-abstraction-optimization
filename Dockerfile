FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .

RUN python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .
