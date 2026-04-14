#!/usr/bin/env bash
# Run preflight, then LIBERO batch benchmarks (extra args go to run_libero_batch after --).
#
# Usage (from repo root):
#   source .venv-libero/bin/activate   # recommended: dedicated LIBERO venv (see docs/libero-tasks.md)
#   ./scripts/benchmark_one_button.sh
#   ./scripts/benchmark_one_button.sh -- --debug
#
# If a venv is already active, `uv run --active` is used so packages match that env.
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="${BENCHMARK_CONFIG:-env_configs/libero/franka_libero_cap_agent0.yaml}"
SERVER_URL="${BENCHMARK_SERVER_URL:-http://127.0.0.1:8110/chat/completions}"
STRICT="${BENCHMARK_PREFLIGHT_STRICT:-1}"

PREFLIGHT_ARGS=(--suite libero --config-path "$CONFIG" --server-url "$SERVER_URL")
if [[ "$STRICT" == "1" || "$STRICT" == "true" ]]; then
  PREFLIGHT_ARGS+=(--strict)
fi

UV_RUN=(uv run)
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  UV_RUN=(uv run --active)
fi

"${UV_RUN[@]}" python scripts/benchmark_preflight.py "${PREFLIGHT_ARGS[@]}"

exec "${UV_RUN[@]}" python capx/envs/scripts/run_libero_batch.py "$@"
