#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "${ROOT_DIR}/scripts/prepare_rog_webqsp_data.py" \
  --output-dir "${ROOT_DIR}/data/rog_webqsp" \
  --chunk-size 1000
