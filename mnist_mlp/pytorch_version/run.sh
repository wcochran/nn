#!/usr/bin/env bash

set -euo pipefail

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

.venv/bin/pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python train_mnist.py
