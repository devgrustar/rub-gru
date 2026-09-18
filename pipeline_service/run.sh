#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_FILE="${CONFIG_PATH:-/workspace/configuration.yaml}"
export CONFIG_FILE

# Raise the open-file limit: the service runs hundreds of threads + Chromium sidecars; a 1024 soft limit
# breaks the validator ("Too many open files") and resets /status connections under load.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || ulimit -n 65536 2>/dev/null || true


# Thread pools: the GLM vLLM's CPU-side image preprocessing spawns nproc-sized OpenMP pools; on a small-quota
# host (10 CPUs seen on RunPod) that oversubscription made the judge's cold path 1.9x slower, while OMP=1 costs
# nothing on 20-40 CPU quotas. Inherited by every vLLM server spawned from llm/spawn.py.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Preflight: network, GPU count, per-GPU bf16 TFLOPS + decode-shaped weight-stream GB/s + power limit.
# Writes /tmp/preflight_metrics.json (picked up by serve.py for the miner-diag header).
if python -m modules.metrics.preflight; then
    echo "=== PRE-FLIGHT OK ==="
else
    echo "=== PRE-FLIGHT FAILED — starting FastAPI in REPLACE mode ==="
fi


# FastAPI
echo "=== STAGE 2: FastAPI ==="
python serve.py &
SERVE_PID=$!


# Dedicated GLM vLLM env (GLM-4.6V needs a newer transformers than the coder).
# Idempotent: only builds when the binary is missing. Non-fatal: on failure the
# judge/critic client is skipped and the coder still comes up.
GLM_VLLM_BIN="${GLM_VLLM_BIN:-/opt/vllm-glm-env/bin/vllm}"
if [ ! -x "$GLM_VLLM_BIN" ]; then
    echo "=== STAGE 2.5: building GLM vLLM env ($GLM_VLLM_BIN missing) ==="
    GLM_MODEL_INFO="$(GLM_VLLM_BIN="$GLM_VLLM_BIN" python - <<'PY'
import os, yaml
try:
    cfg = yaml.safe_load(open(os.environ["CONFIG_FILE"])) or {}
except Exception:
    raise SystemExit(0)
for spec in (cfg.get("llm_clients") or {}).values():
    if not isinstance(spec, dict):
        continue
    v = spec.get("vllm") or {}
    if str(v.get("vllm_bin") or "") == os.environ["GLM_VLLM_BIN"] and str(v.get("model") or "").strip():
        print(str(v["model"]).strip())
        print(str(v.get("revision") or "").strip())
        break
PY
)"
    GLM_MODEL="$(sed -n 1p <<<"$GLM_MODEL_INFO")"
    GLM_REVISION="$(sed -n 2p <<<"$GLM_MODEL_INFO")"
    [ -n "$GLM_MODEL" ] && echo "[run.sh] GLM env build target: $GLM_MODEL @ ${GLM_REVISION:-main}"
    MODEL="$GLM_MODEL" MODEL_REVISION="$GLM_REVISION" \
        bash "$SCRIPT_DIR/scripts/setup_glm_vllm_env.sh" \
        || echo "[run.sh] GLM env setup failed — judge/critic vLLM skipped, coder continues" >&2
else
    echo "[run.sh] GLM vLLM env present at $GLM_VLLM_BIN — skipping build"
fi


# Dedicated coder vLLM env. Same idea as the GLM env above: config-driven and
# idempotent. It builds when missing and upgrades when the requested version
# changes. Non-fatal: failures are logged.
CODER_VLLM_BIN="${CODER_VLLM_BIN:-/opt/vllm-coder-env/bin/vllm}"
# DFlash2 model support landed in vLLM 0.28.0. Keep the coder in its
# dedicated vLLM environment so the GLM server remains independently pinned.
CODER_VLLM_VERSION="${CODER_VLLM_VERSION:-0.28.0}"
CODER_VLLM_INSTALLED_VERSION=""
if [ -x "$CODER_VLLM_BIN" ]; then
    CODER_VLLM_INSTALLED_VERSION="$($CODER_VLLM_BIN --version 2>/dev/null | awk '{print $NF}' || true)"
fi
if [ "$CODER_VLLM_INSTALLED_VERSION" != "$CODER_VLLM_VERSION" ]; then
    [ -n "$CODER_VLLM_INSTALLED_VERSION" ] \
        && echo "=== STAGE 2.6: upgrading coder vLLM $CODER_VLLM_INSTALLED_VERSION -> $CODER_VLLM_VERSION ===" \
        || echo "=== STAGE 2.6: building coder vLLM env $CODER_VLLM_VERSION ($CODER_VLLM_BIN missing) ==="
    CODER_MODEL_INFO="$(GLM_VLLM_BIN="$CODER_VLLM_BIN" python - <<'PY'
import os, yaml
try:
    cfg = yaml.safe_load(open(os.environ["CONFIG_FILE"])) or {}
except Exception:
    raise SystemExit(0)
for spec in (cfg.get("llm_clients") or {}).values():
    if not isinstance(spec, dict):
        continue
    v = spec.get("vllm") or {}
    if str(v.get("vllm_bin") or "") == os.environ["GLM_VLLM_BIN"] and str(v.get("model") or "").strip():
        print(str(v["model"]).strip())
        print(str(v.get("revision") or "").strip())
        break
PY
)"
    CODER_MODEL="$(sed -n 1p <<<"$CODER_MODEL_INFO")"
    CODER_REVISION="$(sed -n 2p <<<"$CODER_MODEL_INFO")"
    if [ -n "$CODER_MODEL" ]; then
        echo "[run.sh] coder env build target: $CODER_MODEL @ ${CODER_REVISION:-main}"
        VENV="$(dirname "$(dirname "$CODER_VLLM_BIN")")" \
        VLLM_VERSION="$CODER_VLLM_VERSION" \
        TORCH_BACKEND="cu129" \
        MODEL="$CODER_MODEL" MODEL_REVISION="$CODER_REVISION" \
            bash "$SCRIPT_DIR/scripts/setup_glm_vllm_env.sh" \
            || echo "[run.sh] coder env setup failed — the coder cannot start without $CODER_VLLM_BIN; check logs" >&2
    fi
else
    echo "[run.sh] coder vLLM env present at $CODER_VLLM_BIN — skipping build"
fi


# vLLM spawn

echo "=== STAGE 3: vLLM spawn ==="
python -m llm.spawn || echo "[run.sh] vllm spawn returned non-zero — FastAPI continues for diagnostics" >&2

wait $SERVE_PID
