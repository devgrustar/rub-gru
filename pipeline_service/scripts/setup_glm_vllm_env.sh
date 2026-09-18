set -euo pipefail

VENV="${VENV:-/opt/vllm-glm-env}"
PYBIN="${PYBIN:-python3.11}"
VLLM_VERSION="${VLLM_VERSION:-0.23.0}"
VLLM_CUDA_TAG="${VLLM_CUDA_TAG:-cu129}"
MANYLINUX="${MANYLINUX:-manylinux_2_28}"
TORCH_BACKEND="${TORCH_BACKEND:-cu128}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=5.0.0rc0}"
FASTAPI_SPEC="${FASTAPI_SPEC:-fastapi<0.137}"
MODEL="${MODEL:-zai-org/GLM-4.6V-Flash}"
MODEL_REVISION="${MODEL_REVISION:-}"

ARCH="$(uname -m)"
WHEEL_URL="${WHEEL_URL:-https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${VLLM_CUDA_TAG}-cp38-abi3-${MANYLINUX}_${ARCH}.whl}"

echo "[glm-env] $VENV | vllm ${VLLM_VERSION}+${VLLM_CUDA_TAG} | torch=$TORCH_BACKEND | $TRANSFORMERS_SPEC"
echo "[glm-env] wheel: $WHEEL_URL"
"$PYBIN" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip uv

if ! "$VENV/bin/uv" pip install --python "$VENV/bin/python" \
        "vllm @ ${WHEEL_URL}" "$TRANSFORMERS_SPEC" "$FASTAPI_SPEC" --torch-backend "$TORCH_BACKEND"; then
    echo "[glm-env] FAIL installing $WHEEL_URL" >&2
    echo "  Check the real asset names for your version:" >&2
    echo "    curl -s https://api.github.com/repos/vllm-project/vllm/releases/tags/v${VLLM_VERSION} | grep -o '[^\"]*\\.whl'" >&2
    echo "  then set VLLM_CUDA_TAG / MANYLINUX / VLLM_VERSION (or WHEEL_URL) accordingly." >&2
    exit 1
fi

HF_TOKEN="${HF_TOKEN:-}" MODEL="$MODEL" MODEL_REVISION="$MODEL_REVISION" "$VENV/bin/python" - <<'PY'
import os, torch, vllm, fastapi
from transformers import AutoProcessor, __version__ as tv
from transformers.processing_utils import ProcessorMixin
torch.zeros(1).cuda()  # raises if the torch/vllm CUDA build mismatches the driver
assert tuple(int(x) for x in fastapi.__version__.split(".")[:2]) < (0, 137), \
    f"fastapi {fastapi.__version__} >= 0.137 -> prometheus /health 500; pin fastapi<0.137"
proc = AutoProcessor.from_pretrained(os.environ["MODEL"], trust_remote_code=True,
                                     revision=os.environ.get("MODEL_REVISION") or None,
                                     token=os.environ.get("HF_TOKEN") or None)
assert isinstance(proc, ProcessorMixin), f"{type(proc).__name__}: transformers too old / missing processor config"
print(f"[glm-env] OK — vllm {vllm.__version__} | torch {torch.__version__} | "
      f"transformers {tv} | fastapi {fastapi.__version__} | {type(proc).__name__}")
PY

echo "[glm-env] done -> configuration.yaml judge-critic-instance.vllm.vllm_bin: $VENV/bin/vllm"