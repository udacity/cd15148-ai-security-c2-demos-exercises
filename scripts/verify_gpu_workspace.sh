#!/usr/bin/env bash
#
# verify_gpu_workspace.sh — confirm the C2 GPU workspace runs every module end to end.
#
# Checks, in order:
#   1. Python and pinned package versions
#   2. Torch device (expects CUDA in the classroom workspace)
#   3. The external asset cache named by $C2_ASSET_CACHE
#   4. Each of the eight notebooks (4 demo + 4 exercise solution), executed with
#      nbconvert exactly the way the module READMEs prescribe
#
# Usage:
#   bash scripts/verify_gpu_workspace.sh                  # everything
#   bash scripts/verify_gpu_workspace.sh --skip-notebooks # checks 1-3 only, ~10 s
#   bash scripts/verify_gpu_workspace.sh --only module-15 # one module
#   bash scripts/verify_gpu_workspace.sh --timeout 3600   # per-notebook cap (default 2700 s)
#   bash scripts/verify_gpu_workspace.sh --allow-missing-cache  # run notebooks even if the
#                                                              # cache check failed (they will download)
#
# A failed cache check skips the notebook phase by default, because the modules
# would download into the sidecar and mask the fault.
#
# Exits 0 only if every selected check passes. Per-notebook logs land in
# $LOG_DIR (default: ./verify-logs).

set -uo pipefail   # deliberately not -e: a failing check must not abort the run

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/verify-logs}"
TIMEOUT_SECONDS=2700
RUN_NOTEBOOKS=1
ALLOW_MISSING_CACHE=0
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-notebooks) RUN_NOTEBOOKS=0; shift ;;
    --allow-missing-cache) ALLOW_MISSING_CACHE=1; shift ;;
    --only)           ONLY="$2"; shift 2 ;;
    --timeout)        TIMEOUT_SECONDS="$2"; shift 2 ;;
    -h|--help)        sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${LOG_DIR}"
PASSES=(); FAILURES=(); SKIPS=()

pass() { PASSES+=("$1");   printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail() { FAILURES+=("$1"); printf '  \033[31mFAIL\033[0m  %s\n' "$1"; }
skip() { SKIPS+=("$1");    printf '  \033[33mSKIP\033[0m  %s\n' "$1"; }
head2() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# Run a command with a wall-clock cap, portable across GNU/BSD (macOS has neither
# `timeout` nor `gtimeout` by default).
run_capped() {
  local seconds="$1" log="$2"; shift 2
  ( "$@" >>"${log}" 2>&1 ) &
  local pid=$!
  ( sleep "${seconds}"; kill -0 "${pid}" 2>/dev/null && kill -9 "${pid}" 2>/dev/null ) &
  local watcher=$!
  wait "${pid}"; local status=$?
  kill -9 "${watcher}" 2>/dev/null; wait "${watcher}" 2>/dev/null
  return "${status}"
}

# ---------------------------------------------------------------- 1. environment

head2 "1. Environment"

"${PYTHON}" - <<'PY'
import sys
major, minor = sys.version_info[:2]
print(f"  python {sys.version.split()[0]}")
sys.exit(0 if (major, minor) == (3, 12) else 1)
PY
[[ $? -eq 0 ]] && pass "Python 3.12" || fail "Python 3.12 (see version above; the GPU image pins 3.12.13)"

# Validate against requirements-gpu.txt itself, so this check cannot drift from
# the file the workspace is built with.
"${PYTHON}" - "${REPO_ROOT}/requirements-gpu.txt" <<'PY'
import importlib, re, sys
from pathlib import Path

# Distribution name -> import name, where the two differ.
IMPORT_NAME = {
    "adversarial-robustness-toolbox": "art",
    "scikit-learn": "sklearn",
    "pillow": "PIL",
}

requirements = Path(sys.argv[1])
if not requirements.exists():
    print(f"  {requirements} not found -- root files missing from this checkout?")
    sys.exit(1)

problems = []
for line in requirements.read_text().splitlines():
    line = line.split("#")[0].strip()
    if not line:
        continue
    match = re.match(r"^([A-Za-z0-9_.\-]+)==([\w.]+)$", line)
    if not match:
        problems.append(f"unparsed requirement line: {line!r}")
        continue
    dist, want = match.groups()
    module = IMPORT_NAME.get(dist.lower(), dist.replace("-", "_"))
    try:
        got = getattr(importlib.import_module(module), "__version__", "?")
    except Exception as exc:
        print(f"  {dist:<32} MISSING ({exc.__class__.__name__})")
        problems.append(f"{dist} not importable")
        continue
    # Compare only the public version. On the CUDA host torch and torchvision
    # report a PEP 440 local version -- "2.5.1+cu121" -- where the "+cu121" label
    # identifies the build, not the release; requirements-gpu.txt pins the release
    # and says so in its own header. A raw `got == want` therefore fails on every
    # correctly built GPU workspace, which makes this whole check unable to pass
    # in the one environment it exists to validate.
    ok = got.split("+", 1)[0] == want
    print(f"  {dist:<32} {got:<14} {'' if ok else f'expected {want}'}")
    if not ok:
        problems.append(f"{dist} {got} != {want}")

sys.exit(1 if problems else 0)
PY
[[ $? -eq 0 ]] && pass "Every pin in requirements-gpu.txt is installed at that version" \
               || fail "requirements-gpu.txt mismatch (see list above)"

"${PYTHON}" - <<'PY'
import sys, torch
if torch.cuda.is_available():
    print(f"  device cuda      {torch.cuda.get_device_name(0)}")
    sys.exit(0)
print(f"  device {'mps' if torch.backends.mps.is_available() else 'cpu'}       no CUDA device visible")
sys.exit(1)
PY
[[ $? -eq 0 ]] && pass "CUDA device available" \
               || fail "No CUDA device — the workspace must expose a GPU (modules still run, far slower)"

# -------------------------------------------------------------- 2. asset cache

head2 "2. External asset cache"

CACHE_OK=0
if [[ -z "${C2_ASSET_CACHE:-}" ]]; then
  fail "C2_ASSET_CACHE is not set — modules will download ~2.4 GB at run time"
elif [[ ! -d "${C2_ASSET_CACHE}" ]]; then
  fail "C2_ASSET_CACHE points at a missing directory: ${C2_ASSET_CACHE}"
else
  echo "  C2_ASSET_CACHE=${C2_ASSET_CACHE}"
  # No --cache-dir: the builder defaults to $C2_ASSET_CACHE, so this exercises
  # the same lookup the module code performs.
  if "${PYTHON}" "${REPO_ROOT}/scripts/build_asset_cache.py" --verify-only; then
    pass "Asset cache is complete"
    CACHE_OK=1
  else
    fail "Asset cache is incomplete (see the manifest above)"
  fi
fi

# ----------------------------------------------------------------- 3. notebooks

# module-dir | prep command (may be empty) | notebook filename
TARGETS=(
  "module-7-apply-evasion-attacks/demo|bash scripts/download_gtsrb_model.sh|adversarial_stop_sign_attack_pipeline.ipynb"
  "module-7-apply-evasion-attacks/exercise/solution|${PYTHON} scripts/prepare_airplane_assets.py|aerial_object_adversarial_evaluation_workflow.ipynb"
  "module-9-apply-data-poisoning/demo||poisoned_image_classification_training_pipeline.ipynb"
  "module-9-apply-data-poisoning/exercise/solution|${PYTHON} scripts/prepare_traffic_sign_assets.py|traffic_sign_label_flip_poisoning_assessment.ipynb"
  "module-15-apply-model-inversion/demo||model_inversion_facility_access_demo.ipynb"
  "module-15-apply-model-inversion/exercise/solution||medical_model_inversion_assessment.ipynb"
  "module-19-apply-quantitative-robustness-testing/demo||robustness_evaluation_pipeline_demo.ipynb"
  "module-19-apply-quantitative-robustness-testing/exercise/solution||traffic_sign_robustness_assessment.ipynb"
)

head2 "3. Module execution"

if [[ ${RUN_NOTEBOOKS} -eq 0 ]]; then
  skip "Notebook execution (--skip-notebooks)"
elif [[ ${CACHE_OK} -eq 0 && ${ALLOW_MISSING_CACHE} -eq 0 ]]; then
  # Running anyway would be actively misleading: the module code points at the
  # cache, so a broken cache means each notebook re-downloads into the sidecar --
  # failing confusingly if it is read-only, or quietly pulling ~800 MB and then
  # "passing" if it is writable, which hides the very fault we are testing for.
  skip "Notebook execution — the asset cache check failed; fix that first, or pass --allow-missing-cache"
else
  for target in "${TARGETS[@]}"; do
    IFS='|' read -r module_dir prep notebook <<<"${target}"
    label="${module_dir}"

    if [[ -n "${ONLY}" && "${module_dir}" != *"${ONLY}"* ]]; then
      continue
    fi
    if [[ ! -d "${REPO_ROOT}/${module_dir}" ]]; then
      skip "${label} (not in this checkout)"
      continue
    fi

    log="${LOG_DIR}/$(echo "${module_dir}" | tr '/' '_').log"
    : >"${log}"
    started=$(date +%s)
    printf '  ....  %s\n' "${label}"

    ok=1
    if [[ -n "${prep}" ]]; then
      # shellcheck disable=SC2086
      ( cd "${REPO_ROOT}/${module_dir}" && run_capped "${TIMEOUT_SECONDS}" "${log}" ${prep} ) || ok=0
      [[ ${ok} -eq 1 ]] || fail "${label} — prep step failed (${log})"
    fi

    if [[ ${ok} -eq 1 ]]; then
      ( cd "${REPO_ROOT}/${module_dir}" && run_capped "${TIMEOUT_SECONDS}" "${log}" \
          "${PYTHON}" -m nbconvert --to notebook --execute "notebooks/${notebook}" \
          --output "verify_$(date +%Y%m%d_%H%M%S).ipynb" --output-dir results ) || ok=0
    fi

    elapsed=$(( $(date +%s) - started ))
    if [[ ${ok} -eq 1 ]]; then
      pass "${label} (${elapsed}s)"
    else
      fail "${label} (${elapsed}s, see ${log})"
    fi
  done
fi

# ------------------------------------------------------------------- 4. summary

head2 "Summary"
printf '  %d passed, %d failed, %d skipped\n' "${#PASSES[@]}" "${#FAILURES[@]}" "${#SKIPS[@]}"
if [[ ${#FAILURES[@]} -gt 0 ]]; then
  printf '\n  Failures:\n'
  printf '    - %s\n' "${FAILURES[@]}"
  printf '\n  Logs: %s\n' "${LOG_DIR}"
  exit 1
fi
printf '\n  GPU workspace verified.\n'
exit 0
