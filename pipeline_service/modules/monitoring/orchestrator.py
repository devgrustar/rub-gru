import asyncio
import multiprocessing
import queue

from fastapi import FastAPI
from logger_config import logger
from pipeline.state import MinerState, MinerStatus
from pipeline.task import PipelineTask
from pipeline.generation_pipeline import GenerationPipeline

from pipeline.state import ServiceState

from .module import _checker_worker
from .settings import ProbeResult

import base64
from pathlib import Path

class OrchestratorChecker:
    def __init__(self, settings) -> None:
        self.settings = settings
        self._queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=4)
        self._last: ProbeResult = ProbeResult.starting(
            getattr(settings, "vllm_model_name", "?"), "not yet probed"
        )
        self._process: multiprocessing.Process | None = None

    @property
    def result(self) -> ProbeResult:
        try:
            while True:
                self._last = self._queue.get_nowait()
        except queue.Empty:
            pass
        return self._last

    def start(self) -> None:
        self._process = multiprocessing.Process(
            target=_checker_worker,
            args=(self.settings, self._queue),
            daemon=True,
        )
        self._process.start()

    def stop(self) -> None:
        if self._process.is_alive():
            self._process.terminate() 
            self._process.join(timeout=5)
        if self._process.is_alive():
            logger.warning(f"checker {self._process.pid} didn't die on SIGTERM, killing")
            self._process.kill()  # SIGKILL
            self._process.join(timeout=2)


async def _abort_generation_and_complete(state: MinerState, generation_task: asyncio.Task | None) -> None:
    if generation_task and not generation_task.done():
        generation_task.cancel()
        try:
            await asyncio.wait_for(generation_task, timeout=10)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception:
            logger.exception("generation task raised during abort")
    state.mark_complete()
    

async def _watch_models(state: MinerState, app: FastAPI, pipeline: GenerationPipeline) -> None:
    """Watch the status of the models and update the state accordingly."""
    while True:
        try:
            checkers: dict[str, OrchestratorChecker] = getattr(app.state, "checkers", {}) or {}
            for name, checker in checkers.items():
                state.set_llm_status(name, checker.result.state)

            if state.status == MinerStatus.WARMING_UP and state.models_ready():
                # Coder throughput probe (once): batch wall = coder tokens / tok/s, so a slow host shows
                # up here before any timed batch. Request REPLACE only while replacements remain; with
                # remaining=0 the /status handler forces WARMING_UP again and the probe is not re-run.
                probe_cfg = pipeline.settings.pipeline.coder_probe
                if probe_cfg.enabled and "probe_tps" not in state.diag:
                    tps = await pipeline.run_coder_probe()
                    if tps is not None and probe_cfg.min_tps > 0 and tps < probe_cfg.min_tps:
                        if state.replacements_remaining > 0:
                            logger.warning(
                                f"[Coder probe] {tps:.0f} tok/s < min_tps {probe_cfg.min_tps:.0f} "
                                f"(replacements_remaining={state.replacements_remaining}) — requesting REPLACE"
                            )
                            state.status = MinerStatus.REPLACE
                            await asyncio.sleep(10)
                            continue
                        logger.warning(
                            f"[Coder probe] {tps:.0f} tok/s < min_tps {probe_cfg.min_tps:.0f} but no replacements "
                            f"remain — continuing on this pod"
                        )
                logger.info("[Warmup] running warmup generation")
                warmup_image_path = Path(__file__).parent.parent.parent / "warmup.png"
                warmup_data_url = f"data:image/png;base64,{base64.b64encode(warmup_image_path.read_bytes()).decode()}"  
                warmup_task = PipelineTask(
                    stem="warmup_task",
                    image_url=warmup_data_url,
                    seed=42
                )
                if pipeline.settings.warmup:
                    await pipeline.run_warmup(warmup_task)
                    
                logger.success("[Warmup] completed successfully")
                state.status = MinerStatus.READY
                logger.info(f"models up — ready ({len(checkers)} probed)")

            elif state.status == MinerStatus.GENERATING and not state.models_ready():
                down = [n for n, s in state.llm_status.items() if s != ServiceState.RUNNING]
                if state.replacements_remaining > 1:
                    logger.warning(
                        f"model(s) down ({down}) | replacements_remaining="
                        f"{state.replacements_remaining} — requesting replace"
                    )
                    state.status = MinerStatus.REPLACE
                else:
                    logger.warning(
                        f"model(s) down ({down}) | replacements_remaining="
                        f"{state.replacements_remaining} (<=1) — keeping pod, "
                        f"completing partial results"
                    )
                    await _abort_generation_and_complete(state, app.state.generation_task)

        except Exception:
            logger.exception("_watch_models iteration failed")

        await asyncio.sleep(10)