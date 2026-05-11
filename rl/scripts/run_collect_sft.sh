#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f "${ROOT_DIR}/rl/.venv-rl/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/rl/.venv-rl/bin/activate"
fi

case "${MODE}" in
  smoke)
    python -m rl.data.collect_sft_dataset --config "${ROOT_DIR}/rl/configs/sft_collection.yaml" --smoke
    ;;
  pilot)
    python -m rl.data.collect_sft_dataset --config "${ROOT_DIR}/rl/configs/sft_collection.yaml" --pilot
    ;;
  full)
    python -m rl.data.collect_sft_dataset --config "${ROOT_DIR}/rl/configs/sft_collection.yaml" --full
    ;;
  resume)
    python -m rl.data.collect_sft_dataset --config "${ROOT_DIR}/rl/configs/sft_collection.yaml" --full --resume
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Usage: bash rl/scripts/run_collect_sft.sh [smoke|pilot|full|resume]"
    exit 1
    ;;
esac

