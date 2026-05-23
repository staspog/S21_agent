#!/usr/bin/env bash
# Minimal venv for build_rag_index.py on the deploy host (faiss + sentence-transformers).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AGENT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV="$ROOT/.rag_build_venv"
MARKER="$VENV/.deps_ok"

_need_reinstall=0
if [[ ! -x "$VENV/bin/python" ]]; then
  _need_reinstall=1
elif [[ ! -f "$MARKER" ]]; then
  _need_reinstall=1
elif [[ -f "$ROOT/requirements.txt" ]] && [[ "$ROOT/requirements.txt" -nt "$MARKER" ]]; then
  _need_reinstall=1
fi

if [[ "$_need_reinstall" = "1" ]]; then
  echo "[ensure_rag_build_venv] Creating $VENV (torch CPU + faiss-cpu + sentence-transformers)..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0"
  "$VENV/bin/pip" install faiss-cpu==1.12.0 sentence-transformers numpy
  touch "$MARKER"
  echo "[ensure_rag_build_venv] Ready."
fi

printf '%s\n' "$VENV/bin/python"
