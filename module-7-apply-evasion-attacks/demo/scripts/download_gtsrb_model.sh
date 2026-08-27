#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Shared workspace asset cache (C2_ASSET_CACHE); falls back to the module's own
# downloads/ folder, so a plain git clone behaves exactly as before.
if [[ -n "${C2_ASSET_CACHE:-}" ]]; then
  MODEL_DIR="${C2_ASSET_CACHE}/hf/gtsrb_vit"
else
  MODEL_DIR="${DEMO_ROOT}/downloads/gtsrb_model"
fi
MODEL_REPO="kelvinandreas/vit-traffic-sign-GTSRB"
BASE_URL="https://huggingface.co/${MODEL_REPO}/resolve/main"

mkdir -p "${MODEL_DIR}"

download_file() {
  local filename="$1"
  local url="${BASE_URL}/${filename}"
  local output="${MODEL_DIR}/${filename}"

  if [[ -s "${output}" ]]; then
    echo "${filename} already present, skipping."
    return
  fi

  echo "Downloading ${filename}..."
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "${output}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${output}" "${url}"
  else
    echo "Error: install curl or wget, then run this script again." >&2
    exit 1
  fi
}

download_file "config.json"
download_file "preprocessor_config.json"
download_file "model.safetensors"

cat <<EOF

Done. Downloaded ${MODEL_REPO} to:
${MODEL_DIR}

You can now run:
python -m nbconvert --to notebook --execute notebooks/adversarial_stop_sign_attack_pipeline.ipynb --output executed_adversarial_stop_sign_attack_pipeline.ipynb --output-dir results
EOF
