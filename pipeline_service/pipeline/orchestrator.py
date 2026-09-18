from __future__ import annotations

import asyncio
import time

import httpx

from llm.session_store import SessionStore
from logger_config import logger
from pipeline.stages import (
    code_and_check,
    critic_stage,
    multigen_first_iter,
    prepare_inputs_stage,
    renderer_stage,
    StageError,
)
from pipeline.task import PipelineTask


class Pipeline:
    """Linear pipeline: prepare-inputs (planner optional), then iterate
    coder→checker→renderer→critic. Iteration 0 may use multi-generation +
    judge bracket when `coder_ensemble_size > 1`. Picks the best-scoring
    snapshot across iterations."""

    def __init__(
        self,
        *,
        planner,
        coder,
        critic,
        judge,
        js_checker,
        renderer,
        session_store: SessionStore,
        embedder=None,
        http_client: httpx.AsyncClient,
        max_iter: int = 2,
        score_threshold: float = 0.8,
        task_deadline_s: float = 60.0,
        coder_multimodal: bool = False,
        use_planner: bool = True,
        coder_ensemble_size: int = 1,
        coder_ensemble_temperature: float = 0.3,
        render_from_object: bool = False,
        seed_offset: int = 0,
        refinement_enabled: bool = True,
        planner_limit: int = 2,
        coder_limit: int = 2,
        renderer_limit: int = 1,
        js_checker_limit: int = 2,
        critic_limit: int = 3,
        judge_limit: int = 4,
    ) -> None:
        self.planner = planner
        self.coder = coder
        self.critic = critic
        self.judge = judge
        self.embedder = embedder
        self.js_checker = js_checker
        self.renderer = renderer
        self.session_store = session_store
        self.http_client = http_client
        self.coder_multimodal = coder_multimodal
        self.use_planner = use_planner
        self.coder_ensemble_size = coder_ensemble_size
        self.coder_ensemble_temperature = coder_ensemble_temperature
        self.render_from_object = render_from_object
        self.seed_offset = seed_offset
        self.refinement_enabled = refinement_enabled

        self.max_iter = max_iter
        self.score_threshold = score_threshold
        self.task_deadline_s = task_deadline_s

        self._sem = {
            "planner": asyncio.Semaphore(planner_limit),
            "coder": asyncio.Semaphore(coder_limit),
            "renderer": asyncio.Semaphore(renderer_limit),
            "critic": asyncio.Semaphore(critic_limit),
            "js_checker": asyncio.Semaphore(js_checker_limit),
            "judge": asyncio.Semaphore(judge_limit),
        }
        self.task_status: dict[str, str] = {}

    def _should_stop(self, report, iteration: int, best_score: float) -> bool:
        if report.stop:
            return True
        if report.overall_score >= self.score_threshold:
            return True
        if not report.issues:
            return True
        if iteration >= self.max_iter:
            return True
        if iteration >= 1 and best_score < 0.20:
            return True
        return False

    async def _run(self, task: PipelineTask) -> None:

        # Planner with downloading image
        await prepare_inputs_stage(
            task,
            planner=self.planner,
            use_planner=self.use_planner,
            http_client=self.http_client,
            sem=self._sem["planner"],
            status=self.task_status,
        )

        last_report = None

        for iteration in range(self.max_iter + 1):
            task.iteration = iteration

            if iteration == 0 and self.coder_ensemble_size > 1:
        
                # First iteration with multigen
                await multigen_first_iter(
                    task,
                    coder=self.coder,
                    judge=self.judge,
                    embedder=self.embedder,
                    js_checker=self.js_checker,
                    renderer=self.renderer,
                    session_store=self.session_store,
                    sem_coder=self._sem["coder"],
                    sem_checker=self._sem["js_checker"],
                    sem_renderer=self._sem["renderer"],
                    sem_judge=self._sem["judge"],
                    coder_multimodal=self.coder_multimodal,
                    status=self.task_status,
                    ensemble_size=self.coder_ensemble_size,
                    ensemble_temperature=self.coder_ensemble_temperature,
                    render_from_object=self.render_from_object,
                    seed_offset=self.seed_offset,
                )
            else:
                # Next iterations without multigen (base coder, patcher and repair agent)
                await code_and_check(
                    task,
                    coder=self.coder,
                    js_checker=self.js_checker,
                    sem_coder=self._sem["coder"],
                    sem_checker=self._sem["js_checker"],
                    coder_multimodal=self.coder_multimodal,
                    status=self.task_status,
                    last_report=last_report,
                    render_from_object=self.render_from_object,
                )

                # Renderer
                await renderer_stage(
                    task,
                    renderer=self.renderer,
                    sem=self._sem["renderer"],
                    status=self.task_status,
                )

            if not self.refinement_enabled:
                # Refinement disabled: keep iteration-0 generation as the final
                # output and skip the critic + all patcher/repair iterations.
                task.best_js_code = task.js_code
                task.best_rendered_png = task.rendered_png
                task.best_iter = iteration
                logger.info(
                    f"[NO-REFINE] {task.stem} | refinement loop disabled — "
                    f"critic + patcher skipped"
                )
                break

            report = await critic_stage(
                task,
                critic=self.critic,
                sem=self._sem["critic"],
                status=self.task_status,
            )

            task.score_history.append(report.overall_score)
            if report.overall_score > task.best_score:
                task.best_score = report.overall_score
                task.best_iter = iteration
                task.best_js_code = task.js_code
                task.best_rendered_png = task.rendered_png

            last_report = report

            if self._should_stop(report, iteration, task.best_score):
                break

        if task.best_js_code is not None:
            task.js_code = task.best_js_code
            task.rendered_png = task.best_rendered_png
           

    async def run_one(self, task: PipelineTask) -> PipelineTask:
        task.started_at = time.time()
        self.task_status[task.stem] = "pending"

        try:
            async with asyncio.timeout(self.task_deadline_s):
                await self._run(task)

            task.failed = False
            task.failure_reason = None
            task.failure_stage = None
            self.task_status[task.stem] = "completed"

        except TimeoutError:
            task.failed = True
            task.failure_reason = f"budget exceeded ({self.task_deadline_s}s)"
            task.failure_stage = "deadline"
            self.task_status[task.stem] = "failed"
            logger.warning(f"[DEADLINE] {task.stem} exceeded {self.task_deadline_s}s")

        except StageError as exc:
            task.failed = True
            task.failure_reason = str(exc)
            task.failure_stage = exc.stage
            self.task_status[task.stem] = "failed"
            logger.exception(f"[FAIL] {task.stem} stage={exc.stage}")

        except Exception as exc:
            task.failed = True
            task.failure_reason = f"unexpected: {type(exc).__name__}: {exc}"
            task.failure_stage = "unknown"
            self.task_status[task.stem] = "failed"
            logger.exception(f"[FAIL] {task.stem} unexpected: {exc}")

        finally:
            elapsed = time.time() - task.started_at
            logger.success(
                f"[TASK COMPLETED] {task.stem} | Scores: {task.score_history} | Best Score: {task.best_score:.2f} | Best Iter: {task.best_iter} | Elapsed: {elapsed:.1f}s"
            )

        return task

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def submit(self, task: PipelineTask) -> asyncio.Task[PipelineTask]:
        return asyncio.create_task(self.run_one(task))
