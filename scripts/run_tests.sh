#!/bin/bash
# Create an isolated environment and validate the package.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo "== creating .venv"
  python3 -m venv .venv
fi
echo "== installing (editable)"
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e .

echo "== validation"
.venv/bin/python tests/validate.py
