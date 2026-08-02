#!/usr/bin/env bash
# Capture the three live server venvs into committed lock files
# (requirements-sd15.lock, requirements-flux1.lock, requirements-stylegan3.lock).
#
# Run this ON THE SERVER, from the repository root (/pitsec_sose26_topic8), whenever a venv's
# installed packages change and the change should become part of the reproducible handover.
# This does not install or modify anything; it only reads each venv's `pip freeze` output.
#
# Usage:
#   bash scripts/capture_env_locks.sh
#   git add requirements-*.lock && git commit -m "chore: refresh venv lock files"
set -euo pipefail

cd "$(dirname "$0")/.."

declare -A VENVS=(
  [sd15]="venv_sd15"
  [flux1]="venv_flux1"
  [stylegan3]="venv_stylegan3"
)

for name in "${!VENVS[@]}"; do
  venv_dir="${VENVS[$name]}"
  py="$venv_dir/bin/python"
  lock="requirements-$name.lock"
  if [ ! -x "$py" ]; then
    echo "[skip] $venv_dir not found or not executable at $py" >&2
    continue
  fi
  "$py" -m pip freeze > "$lock"
  # Local/editable installs (e.g. "-e /home/<user>/...") only work on the exact machine that
  # created the venv; a lock file that contains one is not portable to a fresh container and
  # must be resolved (pin a real version, or document the path as a manual post-install step)
  # before committing.
  if grep -qE '^-e |^ *file://' "$lock"; then
    echo "[REVIEW REQUIRED] $lock contains a local/editable install; fix before committing:" >&2
    grep -nE '^-e |^ *file://' "$lock" >&2
  else
    echo "[ok] wrote $lock ($(wc -l < "$lock") packages)"
  fi
done

echo
echo "Next: review each requirements-*.lock (especially any [REVIEW REQUIRED] above), then:"
echo "  git add requirements-*.lock && git commit -m \"chore: refresh venv lock files\""
