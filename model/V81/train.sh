#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PY="${PYTHON:-/autodl-fs/data/miniconda3/bin/python}"
export MX_THREADS="${MX_THREADS:-3}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" -u analysis.py --pipeline --jobs "${JOBS:-8}" "$@"
