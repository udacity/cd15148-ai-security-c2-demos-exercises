#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXERCISE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${EXERCISE_ROOT}/scripts/prepare_airplane_assets.py"
