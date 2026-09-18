
from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict, field

# Legacy knobs kept for env compatibility; the bf16 benchmark below uses fixed shapes.
MATRIX_SIZE = int(os.environ.get("BENCHMARK_MATRIX_SIZE", "8192"))
DURATION_SEC = float(os.environ.get("BENCHMARK_DURATION_SEC", "3.0"))
STREAM_GB = float(os.environ.get("BENCHMARK_STREAM_GB", "2.0"))
STREAM_SEC = float(os.environ.get("BENCHMARK_STREAM_SEC", "2.0"))

_AUTO_GPU_TOKENS = {"", "auto", "all"}

# Helper functions
def _default_gpu_count() -> int:
    """Return GPU count from env override, or auto-detect via torch.
    """
    if "BENCHMARK_GPU_COUNT" in os.environ:
        return int(os.environ["BENCHMARK_GPU_COUNT"])
    try:
        import torch
        n = torch.cuda.device_count()
        return n if n > 0 else 1
    except Exception:
        return 1

def _detect_gpu_ids_via_nvidia_smi() -> list[str] | None:
    """Return list of GPU indices visible to ``nvidia-smi``, or ``None`` on failure.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        indices = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
        return indices or None
    except Exception:
        return None


def _detect_gpu_ids_via_torch() -> list[str] | None:
    try:
        import torch
        n = torch.cuda.device_count()
        return [str(i) for i in range(n)] if n > 0 else None
    except Exception:
        return None


def _detect_all_gpu_ids() -> list[str]:
    """Auto-detect visible GPUs (nvidia-smi first, torch fallback). Defaults to ``["0"]``."""
    return (
        _detect_gpu_ids_via_nvidia_smi()
        or _detect_gpu_ids_via_torch()
        or ["0"]
    )


def _largest_power_of_two_leq(n: int) -> int:
    """Return the largest power of two less than or equal to n."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def resolve_gpu_ids(gpu_ids: str | None) -> str:
    """Expand ``"all"`` / ``"auto"`` / empty / ``None`` to ``"0,1,...,N-1"``.
    """
    token = (gpu_ids or "").strip().lower()
    if token not in _AUTO_GPU_TOKENS:
        return gpu_ids  # type: ignore[return-value]
    return ",".join(_detect_all_gpu_ids())


# Resolve GPU configuration for vLLM ( only used for restarting vLLM if it is not responding)
def resolve_vllm_gpu_config(
    gpu_ids: str | None,
    tensor_parallel_size: int | None,
) -> tuple[str, int]:
    """Resolve ``(gpu_ids_csv, tensor_parallel_size)`` for a vLLM endpoint.
    """
    resolved_ids = resolve_gpu_ids(gpu_ids)
    n = len([x for x in resolved_ids.split(",") if x.strip()])

    if tensor_parallel_size is None or tensor_parallel_size <= 0:
        tp = max(1, n)
        if os.environ.get("VLLM_TP_POWER_OF_TWO", "").strip() in ("1", "true", "yes"):
            tp = _largest_power_of_two_leq(tp)
    else:
        tp = int(tensor_parallel_size)

    print(
        f"[gpu-resolve] GPU IDs: {resolved_ids!r} | Tensor Parallel Size: {tp} "
        f"(Input GPU IDs: {gpu_ids!r}, Input Tensor Parallel Size: {tensor_parallel_size!r})",
        file=sys.stderr,
        flush=True,
    )
    return resolved_ids, tp


# Benchmark
#
# Why these three numbers (measured 2026-09-05 on an H100 PCIe, 300 W vs 200 W power cap; real coder tok/s fell 22 %):
#   - raw copy bandwidth did not move at all (1850 -> 1847 GB/s): a "GPU bandwidth" check passes a power-throttled host,
#   - bf16 matmul TFLOPS fell 39 % (365 -> 222): reacts, but overshoots,
#   - a decode-shaped GEMM that streams weights x[M,K] @ W[K,N] with M = running seqs x (MTP acceptance+1) brackets the real
#     drop: M=48 -14 %, M=144 -33 %.  That is what a bandwidth/compute-bound decode step looks like, without any model download.
# Thresholds are per GPU family (env override) and deliberately loose until calibrated on a healthy 4xH200 pod
# (run `python -m modules.metrics.gpu` there and read the numbers).

@dataclass
class GPUBenchmarkResult:
    gpu_id: int
    gpu_name: str
    vram_gb: float
    tflops_bf16: float
    stream48_gbps: float
    stream144_gbps: float
    power_limit_w: float | None
    sm_clock_mhz: int | None
    passed: bool
    reasons: list[str] = field(default_factory=list)
    # legacy alias so old log readers keep working
    @property
    def tflops(self) -> float:
        return self.tflops_bf16


def gpu_thresholds(gpu_name: str) -> tuple[float, float]:
    """(min bf16 TFLOPS, min M=48 weight-stream GB/s) for this GPU family; env overrides win."""
    name = (gpu_name or "").upper()
    if "H200" in name:
        d = (530.0, 3000.0)     # healthy H200 SXM measured 2026-09-05 (4 GPUs x 3 restarts): 652-669 TFLOPS, stream48 3697-3739 GB/s -> ~-20 %
    elif "B200" in name:
        d = (600.0, 3000.0)
    elif "H100" in name:
        d = (250.0, 1100.0)     # H100 PCIe @300 W: 365 TFLOPS, 1567 GB/s ; @200 W: 222 / 1347
    else:
        d = (30.0, 0.0)         # unknown family: legacy sanity only
    return (
        float(os.environ.get("BENCHMARK_MIN_TFLOPS", d[0])),
        float(os.environ.get("BENCHMARK_MIN_STREAM48_GBPS", d[1])),
    )


def _smi_query(gpu_id: int, fields: str) -> list[str] | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_id), f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None
        return [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
    except Exception:
        return None


def _benchmark_single_gpu(
    gpu_id: int,
    matrix_size: int,
    duration_sec: float,
    stream_gb: float,
    stream_sec: float,
) -> GPUBenchmarkResult:
    """bf16 matmul TFLOPS + decode-shaped weight-stream GB/s (M=48 / M=144) on one GPU."""
    import torch

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    gpu_name = torch.cuda.get_device_name(gpu_id)
    props = torch.cuda.get_device_properties(gpu_id)
    vram_gb = round(props.total_memory / (1024**3), 1)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    K = N = matrix_size
    per_mat = K * N * 2
    n_mat = max(2, int(stream_gb * 1024**3 // per_mat))
    Ws = [torch.randn(K, N, device=device, dtype=torch.bfloat16) for _ in range(n_mat)]

    # 1. compute-bound: sustained bf16 matmul
    a, b = Ws[0], Ws[1]
    for _ in range(3):
        a @ b
    torch.cuda.synchronize(device)
    ops = 0
    start = time.monotonic()
    while time.monotonic() - start < duration_sec:
        a @ b
        torch.cuda.synchronize(device)
        ops += 1
    elapsed = time.monotonic() - start
    tflops = round(ops * 2.0 * K * N * K / elapsed / 1e12, 1)

    # 2. decode-shaped: stream all weights through x[M,K] @ W for M = 48 and 144
    def stream(M: int) -> float:
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        for W in Ws[:2]:
            x @ W
        torch.cuda.synchronize(device)
        t0 = time.monotonic()
        passes = 0
        while time.monotonic() - t0 < stream_sec:
            for W in Ws:
                x @ W
            torch.cuda.synchronize(device)
            passes += 1
        dt = time.monotonic() - t0
        return round(passes * n_mat * per_mat / dt / 1e9, 0)

    s48 = stream(48)
    s144 = stream(144)

    smi = _smi_query(gpu_id, "power.limit,clocks.sm")
    power_limit = float(smi[0]) if smi and smi[0].replace(".", "", 1).isdigit() else None
    sm_clock = int(float(smi[1])) if smi and len(smi) > 1 and smi[1].replace(".", "", 1).isdigit() else None

    min_tflops, min_s48 = gpu_thresholds(gpu_name)
    reasons = []
    if tflops < min_tflops:
        reasons.append(f"bf16 matmul {tflops} TFLOPS < {min_tflops}")
    if s48 < min_s48:
        reasons.append(f"weight-stream M=48 {s48:.0f} GB/s < {min_s48:.0f}")
    min_power = float(os.environ.get("BENCHMARK_MIN_POWER_LIMIT_W", "600" if "H200" in gpu_name.upper() else "0"))
    if power_limit is not None and min_power > 0 and power_limit < min_power:
        reasons.append(f"power.limit {power_limit:.0f} W < {min_power:.0f}")

    del Ws, a, b
    torch.cuda.empty_cache()
    return GPUBenchmarkResult(
        gpu_id=gpu_id, gpu_name=gpu_name, vram_gb=vram_gb, tflops_bf16=tflops,
        stream48_gbps=s48, stream144_gbps=s144, power_limit_w=power_limit, sm_clock_mhz=sm_clock,
        passed=not reasons, reasons=reasons,
    )


def run_benchmark(
    num_gpus: int | None = None,
    matrix_size: int = MATRIX_SIZE,
    duration_sec: float = DURATION_SEC,
    stream_gb: float = STREAM_GB,
    stream_sec: float = STREAM_SEC,
) -> list[GPUBenchmarkResult]:
    """Benchmark all available GPUs (or num_gpus if specified) in parallel"""
    import torch.multiprocessing as mp

    if num_gpus is None:
        num_gpus = _default_gpu_count()

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as pool:
        futures = [
            pool.submit(_benchmark_single_gpu, gpu_id, matrix_size, duration_sec, stream_gb, stream_sec)
            for gpu_id in range(num_gpus)
        ]
        return [f.result() for f in futures]


def results_as_dicts(results: list[GPUBenchmarkResult]) -> list[dict]:
    return [asdict(r) for r in results]


if __name__ == "__main__":
    import json
    print(json.dumps(results_as_dicts(run_benchmark()), indent=1))
