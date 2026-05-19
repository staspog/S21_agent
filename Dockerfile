# Образ для нового приложения: FastAPI + Rocket.Chat search pipeline.
# Локальные индексы и файлы старого пайплайна в образ больше не копируются.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    UVICORN_RELOAD=0

# Минимальные runtime-пакеты (нужны для torch/sentence-transformers и HTTPS).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only PyTorch: на Linux иначе тянется torch+cuda (nvidia_*, triton…) — гигабайты и долгий шаг «Installing collected packages».
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0" \
    && pip install -r requirements.txt

COPY ./src ./src
COPY ./main.py ./main.py
COPY ./faiss_store ./faiss_store

RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/.cache/huggingface \
    && chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

CMD ["python", "main.py"]
