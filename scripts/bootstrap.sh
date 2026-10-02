#!/usr/bin/env bash
set -euo pipefail
# Use repository-local defaults, with explicit overrides for other volumes.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"
if ! command -v uv >/dev/null 2>&1; then
  printf 'Install uv first using https://docs.astral.sh/uv/getting-started/installation/\n' >&2
  exit 1
fi
export FIRERISK_HOME="${FIRERISK_HOME:-$project_root/.artifacts}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$FIRERISK_HOME/uv-cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$project_root/.venv}"
mkdir -p "$FIRERISK_HOME"
uv sync --frozen --extra dev --python 3.12
"$UV_PROJECT_ENVIRONMENT/bin/firerisk" doctor
printf '\nSet artifacts with: export FIRERISK_HOME=%q\n' "$FIRERISK_HOME"
printf 'Activate with: source %q/bin/activate\n' "$UV_PROJECT_ENVIRONMENT"
