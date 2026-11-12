#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e ".[dev]"
vyrex show-config --config configs/base.yaml
vyrex doctor
pytest -q
