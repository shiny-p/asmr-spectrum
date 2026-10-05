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

echo "== environment"
.venv/bin/python -c "import sys, numpy, matplotlib; \
print('python    ', sys.version.split()[0]); \
print('numpy     ', numpy.__version__); \
print('matplotlib', matplotlib.__version__)"

echo "== validation: spectral analysis"
.venv/bin/python tests/validate.py

echo "== validation: parameter audit"
.venv/bin/python tests/validate_audit.py

