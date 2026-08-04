#!/usr/bin/env bash
# Capture exact package snapshots from the three live server environments.
# This script reads environments only. It does not install or modify packages.
set -euo pipefail

cd "$(dirname "$0")/.."

allow_missing=0
if [ "${1:-}" = "--allow-missing" ]; then
  allow_missing=1
elif [ "$#" -ne 0 ]; then
  echo "Usage: bash scripts/capture_env_locks.sh [--allow-missing]" >&2
  exit 2
fi

mkdir -p requirements/locks

sd15_py="${WTP_PY_DEFAKE:-${WTP_PY_SD15:-${WTP_ROOT:-$(pwd)}/venv_sd15/bin/python3}}"
flux1_py="${WTP_PY_FLUX1:-${WTP_ROOT:-$(pwd)}/venv_flux1/bin/python3}"
stylegan3_py="${WTP_PY_STYLEGAN3:-${WTP_ROOT:-$(pwd)}/venv_stylegan3/bin/python3}"

names=("sd15-defake" "flux1" "stylegan3")
interpreters=("$sd15_py" "$flux1_py" "$stylegan3_py")
failed=0

for i in 0 1 2; do
  name="${names[$i]}"
  py="${interpreters[$i]}"
  lock="requirements/locks/${name}.txt"

  if [ ! -x "$py" ]; then
    echo "[missing] $name interpreter is not executable: $py" >&2
    if [ "$allow_missing" -eq 0 ]; then
      failed=1
    fi
    continue
  fi

  tmp="${lock}.tmp"
  "$py" -m pip freeze --all | LC_ALL=C sort > "$tmp"

  if grep -nE '(^-e[[:space:]]|@[[:space:]]+file:|^[^#]+[[:space:]]+@[[:space:]]+/|file://)' "$tmp"; then
    echo "[review required] $name contains a local or editable dependency." >&2
    rm -f "$tmp"
    failed=1
    continue
  fi

  mv "$tmp" "$lock"
  count=$(wc -l < "$lock" | tr -d ' ')
  echo "[ok] wrote $lock ($count entries) from $py"
done

if [ "$failed" -ne 0 ]; then
  echo "Environment-lock capture failed. Resolve the messages above before handover." >&2
  exit 1
fi

cat <<'MSG'

Review the generated files, especially VCS package revisions and CUDA-related packages, then run:
  git diff -- requirements/locks/
  git add requirements/locks/
MSG
