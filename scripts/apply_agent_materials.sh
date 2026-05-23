#!/usr/bin/env bash
# Overlay private agent materials (adm_info, corpus, …) onto AGENT_ROOT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_ROOT="${AGENT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MATERIALS_AGENT_DIR="${MATERIALS_AGENT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)/secrets/materials/agent}"

if [[ ! -d "$MATERIALS_AGENT_DIR" ]]; then
  echo "✘ Materials dir not found: $MATERIALS_AGENT_DIR" >&2
  exit 1
fi

echo "[apply_agent_materials] $MATERIALS_AGENT_DIR/ → $AGENT_ROOT/"
rsync -a "$MATERIALS_AGENT_DIR/" "$AGENT_ROOT/"
