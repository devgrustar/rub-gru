from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
import zipfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse, FileResponse

from config.settings import LLMClientConfig, settings
from logger_config import logger
from pipeline.generation_pipeline import GenerationPipeline
from pipeline.state import MinerState, MinerStatus
from pipeline.task import PipelineTask
from schemas.requests import GenerateRequest
from schemas.responses import GenerateAccepted, StatusResponse
from modules.monitoring import (
    OrchestratorChecker,
    _conflict,
    _watch_models,
)


state = MinerState()
pipeline = GenerationPipeline(settings, state)
_generate_lock = asyncio.Lock()


class _VllmCheckerAdapter:
    """Adapter for the VLLM client.
    """

    def __init__(self, name: str, cfg: LLMClientConfig) -> None:
        if cfg.vllm is None:
            raise ValueError(f"llm_clients.{name}: missing vllm sub-config")
        self.name = name
        self.vllm_model_name = cfg.vllm.model
        self.vllm_url = cfg.base_url
        self.vllm_api_key = cfg.vllm.api_key or "local"


def _build_checkers() -> dict[str, OrchestratorChecker]:
    """
    Build the list of OrchestratorCheckers for the local vLLM clients.
    """
    checkers: dict[str, OrchestratorChecker] = {}
    for name, cfg in settings.llm_clients.items():
        if not cfg.enabled:
            continue
        if cfg.backend != "vllm":
            continue
        if cfg.vllm is None or not (cfg.vllm.model or "").strip():
            continue
        adapter = _VllmCheckerAdapter(name, cfg)
        checker = OrchestratorChecker(adapter)
        checker.start()
        checkers[name] = checker
        logger.info(
            f"OrchestratorChecker started | Client: {name} | "
            f"Model: {adapter.vllm_model_name} | URL: {adapter.vllm_url}"
        )
    return checkers


_PREFLIGHT_METRICS = os.environ.get("PREFLIGHT_METRICS_FILE", "/tmp/preflight_metrics.json")


def _load_preflight_metrics() -> None:
    """Fold /tmp/preflight_metrics.json (written by modules.metrics.preflight) into state.diag."""
    try:
        with open(_PREFLIGHT_METRICS) as f:
            m = json.load(f)
    except Exception:
        return
    host = m.get("host") or {}
    gpus = m.get("gpus") or []
    d = state.diag
    d["preflight_ok"] = m.get("ok")
    d["cpu_quota"] = host.get("cpu_quota")
    d["nproc"] = host.get("nproc")
    d["mem_gb"] = host.get("mem_gb")
    net = m.get("network") or {}
    d["net_mbps"] = net.get("download_mbps")
    if gpus:
        names = sorted({str(g.get("gpu_name", "?")).replace("NVIDIA ", "") for g in gpus})
        d["gpus"] = f"{len(gpus)}x{'/'.join(names)}"
        d["tflops"] = [g.get("tflops_bf16") for g in gpus]
        d["stream48"] = [g.get("stream48_gbps") for g in gpus]
        d["power_w"] = [g.get("power_limit_w") for g in gpus]
    logger.info(f"preflight metrics loaded: {d}")


def _diag_header() -> str:
    """One ASCII comment line describing the host this module was generated on."""
    d = state.diag
    def _lst(v):
        if isinstance(v, list):
            return "/".join("?" if x is None else f"{float(x):.0f}" for x in v)
        return "?" if v is None else str(v)
    wall = int(time.time() - state.batch_started_at) if state.batch_started_at else None
    parts = [
        f"gpus={d.get('gpus', '?')}",
        f"tflops={_lst(d.get('tflops'))}",
        f"stream48={_lst(d.get('stream48'))}",
        f"power={_lst(d.get('power_w'))}",
        f"probe_tps={d.get('probe_tps', '?')}",
        f"cpu_quota={d.get('cpu_quota', '?')}/{d.get('nproc', '?')}",
        f"net={d.get('net_mbps', '?')}",
        f"batch={state.batch_index}",
        f"wall={wall if wall is not None else '?'}",
        f"seed={state.seed}",
    ]
    line = "// miner-diag: " + " ".join(parts)
    return "".join(ch for ch in line if 32 <= ord(ch) < 127) + "\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists("/tmp/pod_replace"):
        logger.warning("pre-flight failed — pod marked for replacement")
        state.status = MinerStatus.REPLACE
    _load_preflight_metrics()

    checkers = _build_checkers()
    if not checkers:
        logger.info("No local vLLM clients in llm_clients — cloud-only mode (no health probes)")

    app.state.checkers = checkers
    app.state.generation_task = None

    await pipeline.startup()
    watch_task = asyncio.create_task(_watch_models(state, app, pipeline))

    try:
        yield
    finally:
        watch_task.cancel()
        try:
            await watch_task
        except asyncio.CancelledError:
            pass
        for name, checker in checkers.items():
            try:
                checker.stop()
            except Exception:
                logger.exception(f"checker stop failed | client={name}")
        await pipeline.shutdown()


app = FastAPI(title="Miner Pipeline", lifespan=lifespan)


@app.get("/health")
async def health():
    return Response(status_code=200)


@app.get("/status", response_model=StatusResponse)
async def status(replacements_remaining: int = 0):
    state.replacements_remaining = replacements_remaining
    if replacements_remaining == 0 and state.status == MinerStatus.REPLACE:
        logger.warning("remaining=0 on REPLACE pod — forcing WARMING_UP to attempt generation")
        state.status = MinerStatus.WARMING_UP
    return state.to_status_response()


@app.post("/generate", response_model=GenerateAccepted)
async def generate(request: GenerateRequest):
    async with _generate_lock:
        if state.status == MinerStatus.WARMING_UP:
            raise HTTPException(503, "Still warming up")

        incoming_stems = sorted(p.stem for p in request.prompts)
        if (
            state.status == MinerStatus.GENERATING
            and sorted(state.batch_stems) == incoming_stems
        ):
            return GenerateAccepted(accepted=len(request.prompts))

        if state.status not in {MinerStatus.READY, MinerStatus.COMPLETE}:
            return _conflict(state, "Cannot accept batch")

        gen_task = app.state.generation_task
        if gen_task and not gen_task.done():
            gen_task.cancel()
            try:
                await gen_task
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        tasks = [
            PipelineTask(stem=p.stem, image_url=p.image_url, seed=request.seed)
            for p in request.prompts
        ]
        state.reset_for_batch([p.stem for p in request.prompts], request.seed)

        async def _run_batch_safe():
            try:
                await pipeline.run_batch(tasks)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("pipeline.run_batch crashed")
            finally:
                if state.status == MinerStatus.GENERATING:
                    state.mark_complete()

        app.state.generation_task = asyncio.create_task(_run_batch_safe())

    return GenerateAccepted(accepted=len(tasks))


@app.get("/results")
async def results():
    if state.status != MinerStatus.COMPLETE:
        raise HTTPException(409, f"Not complete, current status: {state.status.value}")

    header = _diag_header()
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for stem, js_code in state.results.items():
            if not js_code.startswith("// miner-diag:"):
                js_code = header + js_code
            zf.writestr(f"{stem}.js", js_code)
        if state.failed:
            failed_list = [{"stem": s, "reason": r} for s, r in state.failed.items()]
            zf.writestr("_failed.json", json.dumps(failed_list, indent=2))
    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type="application/zip")



def _completed_task_view(t: PipelineTask) -> dict:
    osd_preview = (t.osd[:80] + "…") if t.osd and len(t.osd) > 80 else t.osd
    return {
        "stem": t.stem,
        "status": "completed",
        "failed": t.failed,
        "failure_reason": t.failure_reason,
        "attempt": t.attempt,
        "osd_ok": bool(t.osd),
        "osd_preview": osd_preview,
        "compiler_ok": bool(t.js_code),
        "js_code_bytes": len(t.js_code.encode("utf-8")) if t.js_code else 0,
        "js_valid": t.js_valid,
        "js_errors_count": len(t.js_errors or []),
        "has_png": t.rendered_png is not None,
        "png_bytes": len(t.rendered_png) if t.rendered_png else 0,
        "timing": {"js_total_ms": t.js_total_ms, "render_ms": t.render_ms},
        "metrics": {
            "vertices": (t.js_metrics or {}).get("vertices"),
            "draw_calls": (t.js_metrics or {}).get("drawCalls"),
            "max_depth": (t.js_metrics or {}).get("maxDepth"),
        } if t.js_metrics else None,
        "refinement": {
            "iteration": t.iteration,
            "best_iter": t.best_iter,
            "best_score": t.best_score,
            "score_history": t.score_history,
        },
    }


@app.get("/debug/tasks")
async def debug_tasks():
    tasks_view: list[dict] = []
    rendered = ok = failed = in_progress = 0
    stems = state.batch_stems or list(state.tasks.keys())
    for stem in stems:
        t = state.tasks.get(stem)
        if t is None:
            in_progress += 1
            tasks_view.append({"stem": stem, "status": "in_progress"})
            continue
        if t.rendered_png:
            rendered += 1
        if t.failed:
            failed += 1
        else:
            ok += 1
        tasks_view.append(_completed_task_view(t))
    return {
        "status": state.status.value,
        "progress": state.progress,
        "total": state.total,
        "counts": {"ok": ok, "failed": failed, "in_progress": in_progress, "rendered": rendered},
        "tasks": tasks_view,
    }


@app.get("/debug/tasks/{stem}")
async def debug_task(stem: str):
    task = state.tasks.get(stem)
    if task is None:
        if stem in state.batch_stems:
            return Response(
                status_code=202,
                content=json.dumps({"stem": stem, "status": "in_progress"}),
                media_type="application/json",
            )
        raise HTTPException(404, f"Task '{stem}' not found in current batch")
    return {
        "stem": task.stem,
        "status": "completed",
        "image_url": task.image_url,
        "seed": task.seed,
        "osd": task.osd,
        "js_code": task.js_code,
        "js_valid": task.js_valid,
        "js_errors": task.js_errors,
        "js_metrics": task.js_metrics,
        "js_stages_run": task.js_stages_run,
        "js_module_load_ms": task.js_module_load_ms,
        "js_execution_ms": task.js_execution_ms,
        "js_total_ms": task.js_total_ms,
        "failed": task.failed,
        "failure_reason": task.failure_reason,
        "attempt": task.attempt,
        "render_ms": task.render_ms,
        "render_errors": task.render_errors,
        "rendered_png_b64": (
            base64.b64encode(task.rendered_png).decode() if task.rendered_png else None
        ),
        "multigen_pngs_b64": [
            base64.b64encode(png).decode() if png else None
            for png in (task.multigen_pngs or [])
        ] if task.multigen_pngs else None,
        "refinement_rendered_pngs_b64": [
            base64.b64encode(png).decode() if png else None
            for png in (task.refinement_rendered_pngs or [])
        ] if task.refinement_rendered_pngs else None,
        "refinement": {
            "iteration": task.iteration,
            "best_iter": task.best_iter,
            "best_score": task.best_score,
            "score_history": task.score_history,
        },
        "meta": task.meta,
    }
@app.get("/debug/logs")
async def debug_logs():
    log_path = "logs/pipeline.log"
    if not os.path.exists(log_path):
        raise HTTPException(404,"Log file not found")
    
    return FileResponse(
        path=log_path,
        filename="logs.txt",
        media_type ="text/plain"
    )
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api.host, port=settings.api.port, reload=False)
