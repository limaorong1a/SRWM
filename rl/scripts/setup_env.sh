#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/rl/.venv-rl"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/rl/requirements-rl.txt"

echo "RL collection environment ready at: ${VENV_DIR}"
echo "If ALFWorld is not installed in this env yet, install it now before collection."

