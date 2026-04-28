# Образ для нового приложения: FastAPI + Rocket.Chat search pipeline.
# Локальные индексы и файлы старого пайплайна в образ больше не копируются.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

# Runtime-библиотеки нужны для torch/sentence-transformers и HTTPS-запросов.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY ./src ./src
COPY ./main.py ./main.py

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/.cache/huggingface \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Один воркер: MemorySaver хранит историю в памяти процесса.
# Timeout выше стандартного 30s: RC-поиск + reranker + GigaChat могут отвечать дольше.
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--timeout", "300", "src.api:app"]