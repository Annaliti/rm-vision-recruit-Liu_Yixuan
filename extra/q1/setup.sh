#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="/Users/anna/miniconda3/envs/rmvision/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then PYTHON_BIN="python3"; fi
"$PYTHON_BIN" -m venv "$PROJECT_DIR/.venv"
"$PROJECT_DIR/.venv/bin/python" -m pip install --disable-pip-version-check torch==2.5.1 onnx==1.17.0 onnxruntime==1.20.1 numpy==1.26.4 pillow==10.4.0 matplotlib==3.9.2
"$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/download_dataset.py"
