#!/usr/bin/env bash
# Build faiss_store inside Docker (no host venv / python3-venv required).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AGENT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
IMAGE="${S21_RAG_BUILD_IMAGE:-python:3.11-slim}"

if ! command -v docker >/dev/null 2>&1; then
  echo "✘ docker не найден для build_rag_index_docker.sh" >&2
  exit 1
fi

echo "[build_rag_index_docker] image=$IMAGE root=$ROOT"

docker run --rm \
  -v "$ROOT:/app" \
  -w /app \
  -e AGENT_ROOT=/app \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
  -e MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
  -e TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-2}" \
  -e HF_HOME=/app/.cache/huggingface \
  "$IMAGE" \
  bash -c '
    set -euo pipefail
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      ca-certificates libgomp1 >/dev/null
    pip install -q --upgrade pip
    pip install -q --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0"
    pip install -q faiss-cpu==1.12.0 sentence-transformers numpy
    python scripts/build_rag_index.py --skip-apply
  '
