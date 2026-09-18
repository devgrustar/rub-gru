from __future__ import annotations

import asyncio
import hashlib
import time
from contextlib import asynccontextmanager

import httpx

from logger_config import logger
from modules.scene_planner.schema import OSD
from pipeline.task import Candidate, PipelineTask
from utils.http import download_image


class StageError(Exception):
    """Raised by a pipeline stage; carries stage name + original cause."""

    def __init__(self, stage: str, cause: Exception):
        super().__init__(f"{stage}: {type(cause).__name__}: {cause}")
        self.stage = stage
        self.cause = cause


@asynccontextmanager
async def stage_guard(
    task: PipelineTask,
    stage_name: str,
    sem: asyncio.Semaphore,
    status: dict[str, str],
):
    """Acquire semaphore, log start/done, wrap exceptions in StageError."""
    async with sem:
        status[task.stem] = stage_name
        t0 = time.time()
        try:
            yield
        except Exception as exc:
            raise StageError(stage_name, exc) from exc
        finally:
            dt = time.time() - t0


async def prepare_inputs_stage(
    task: PipelineTask,
    *,
    planner,
    use_planner: bool,
    http_client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    status: dict[str, str],
) -> None:
    """Always download the reference image. Optionally run the planner.

    When `use_planner=False` (or planner=None), `task.osd` stays None and the
    coder/critic must work in pure-image mode (`actors.coder.multimodal=true`).
    """
    async with stage_guard(task, "prepare", sem, status):
        task.image_bytes, task.image_mime = await download_image(
            task.image_url, http_client
        )
        if use_planner and planner is not None:
            osd = await planner.plan(
                task_id=task.stem,
                image_bytes=task.image_bytes,
                image_url=task.image_url,
                mime=task.image_mime,
            )
            task.osd = osd.model_dump_json(indent=2)
        else:
            task.osd = None


async def code_and_check(
    task: PipelineTask,
    *,
    coder,
    js_checker,
    sem_coder: asyncio.Semaphore,
    sem_checker: asyncio.Semaphore,
    coder_multimodal: bool,
    status: dict[str, str],
    last_report=None,
    render_from_object: bool = False,
) -> None:
    """Code → JS check → retry-on-js-error loop. Returns when js_valid is True."""
    checker_mode = "with_object" if render_from_object else "sanity"
    osd = OSD.model_validate_json(task.osd) if task.osd is not None else None
    send_image = (osd is None) or coder_multimodal
    attempt = 0
    while True:
        async with stage_guard(task, "coder", sem_coder, status):
            if task.iteration == 0 and attempt == 0:
                task.js_code = await coder.code(
                    task_id=task.stem,
                    osd=osd,
                    image_bytes=task.image_bytes if send_image else None,
                    image_mime=task.image_mime,
                )
            elif last_report is not None and attempt == 0:
                task.js_code = await coder.code_critic_repair(
                    task_id=task.stem,
                    osd=osd,
                    issues=last_report.issues,
                    overall_score=last_report.overall_score,
                    matching_aspects=list(
                        getattr(last_report, "matching_aspects", []) or []
                    ),
                    image_bytes=task.image_bytes if coder_multimodal else None,
                    image_mime=task.image_mime,
                    render_png=task.rendered_png if coder_multimodal else None,
                )
            else:
                task.js_code = await coder.code_repair(
                    task_id=task.stem,
                    osd=osd,
                    js_errors=list(task.js_errors or []),
                )

        async with stage_guard(task, "js_checker", sem_checker, status):
            task.js_valid = None
            task.js_errors = []
            task.scene_json = None
            await js_checker.process(task, mode=checker_mode)

        if task.js_valid:
            return

        attempt += 1


async def renderer_stage(
    task: PipelineTask,
    *,
    renderer,
    sem: asyncio.Semaphore,
    status: dict[str, str],
) -> None:
    async with stage_guard(task, "renderer", sem, status):
        task.render_errors = []
        task.failed = False
        task.failure_reason = None

        await renderer.process(task)

        if task.failed or task.rendered_png is None:
            reason = task.failure_reason or (
                task.render_errors[0] if task.render_errors else "no png"
            )
            raise StageError("renderer", RuntimeError(reason))


async def critic_stage(
    task: PipelineTask,
    *,
    critic,
    sem: asyncio.Semaphore,
    status: dict[str, str],
):
    async with stage_guard(task, "critic", sem, status):
        if (
            task.image_bytes is None
            or task.js_code is None
            or task.rendered_png is None
        ):
            raise StageError("critic", RuntimeError("missing inputs"))

        osd = OSD.model_validate_json(task.osd) if task.osd is not None else None
        report = await critic.critique(
            task_id=task.stem,
            image_bytes=task.image_bytes,
            image_mime=task.image_mime,
            render_png=task.rendered_png,
            artifact_context={
                "kind": "coder_v1",
                "js_code": task.js_code,
                "osd": osd.model_dump() if osd is not None else None,
            },
        )
        return report


def _code_digest(js_code: str | None) -> str | None:
    """Stable content hash used to fold byte-identical candidate programs onto one."""
    if not js_code:
        return None
    return hashlib.sha256(js_code.encode("utf-8")).hexdigest()


def _is_live(cand: Candidate | None) -> bool:
    """A candidate still in contention: it produced code that rendered cleanly."""
    return (
        cand is not None
        and cand.drop_reason is None
        and cand.rendered_png is not None
    )


async def _resolve_bracket(
    *,
    task: PipelineTask,
    leaf_tasks: dict[int, asyncio.Task],
    judge,
    sem_judge: asyncio.Semaphore,
) -> Candidate | None:
    """Streaming single-elimination over fixed candidate positions.

    Each position is resolved by its own leaf task (coder -> dedup -> render); a
    position that dropped is a walkover for its opponent, costing no judge call.
    A duel fires the moment both of its subtree winners are known, so judging
    overlaps rendering. Pairing is by position, independent of which candidates
    drop, so the outcome is reproducible under a fixed seed.
    """

    async def _judge_duel(left: Candidate, right: Candidate, round_no: int) -> Candidate:
        """Compare two live candidates head-to-head and return the winner."""
        label = f"R{round_no} k{left.k}-vs-k{right.k}"
        _t_q = time.monotonic()
        async with sem_judge:
            _t_s = time.monotonic()
            verdict = await judge.compare(
                task_id=task.stem,
                match_label=label,
                reference_bytes=task.image_bytes,
                reference_mime=task.image_mime,
                render_a=left.rendered_png,
                render_b=right.rendered_png,
                white_views_a=left.judge_white_views,
                white_views_b=right.judge_white_views,
                gray_views_a=left.judge_gray_views,
                gray_views_b=right.judge_gray_views,
                embeddings_a=left.judge_embeddings,
                embeddings_b=right.judge_embeddings,
            )
        winner = left if verdict.winner == "A" else right
        _t_e = time.monotonic()
        logger.info(
            f"[BRACKET {label}] {task.stem} -> k{winner.k} | "
            f"[JUDGE_TIMING] queue_wait={_t_s - _t_q:.1f}s compare={_t_e - _t_s:.1f}s in_flight={sem_judge._value}"
        )
        return winner

    async def _winner_of(lo: int, hi: int) -> Candidate | None:
        """Winner of candidate positions [lo, hi): one leaf, or the better half."""
        if hi - lo <= 1:
            return await leaf_tasks[lo]

        mid = (lo + hi) // 2
        left, right = await asyncio.gather(_winner_of(lo, mid), _winner_of(mid, hi))

        # A half with no live candidate is a walkover — the other side advances free.
        if not _is_live(left):
            return right
        if not _is_live(right):
            return left

        round_no = (hi - lo - 1).bit_length()  # R1 = first-round pair, then upward
        return await _judge_duel(left, right, round_no)

    if not leaf_tasks:
        return None
    return await _winner_of(0, len(leaf_tasks))


def _promote_winner(
    task: PipelineTask,
    winner: Candidate,
    candidates: list[Candidate],
    session_store,
) -> None:
    """Copy the winning candidate onto the task and free the losers' sessions."""
    task.winner_k = winner.k
    task.js_code = winner.js_code
    task.js_valid = winner.js_valid
    task.js_errors = list(winner.js_errors)
    task.scene_json = winner.scene_json
    task.rendered_png = winner.rendered_png

    # Rename the winner's coder session to the canonical actor so downstream
    # repair stages find it; evict the losers to free memory.
    session_store.rename_actor(task.stem, f"coder#k{winner.k}", "coder")
    for cand in candidates:
        if cand.k != winner.k:
            session_store.evict_actor(task.stem, f"coder#k{cand.k}")


class _CandidateFactory:
    """Generates, deduplicates, validates and renders one task's candidates.

    Owns the per-task config and stage dependencies so the orchestrator can fan
    out candidates without threading a dozen arguments through nested closures.
    Each bracket position is resolved by `leaf`: it awaits that position's coder,
    folds byte-identical programs onto their lowest-k leader (seed-deterministic,
    so the bracket is reproducible), then validates the leader through js-check,
    render and judge views. Only leaders render; duplicates and coder failures
    are walkovers.
    """

    def __init__(
        self,
        task: PipelineTask,
        *,
        coder,
        judge,
        embedder,
        js_checker,
        renderer,
        sem_coder: asyncio.Semaphore,
        sem_checker: asyncio.Semaphore,
        sem_renderer: asyncio.Semaphore,
        status: dict[str, str],
        osd: OSD | None,
        send_image: bool,
        checker_mode: str,
        ensemble_temperature: float,
        seed_offset: int = 0,
    ) -> None:
        self.task = task
        self.seed_offset = seed_offset
        self.coder = coder
        self.judge = judge
        self.embedder = embedder
        self.js_checker = js_checker
        self.renderer = renderer
        self.sem_coder = sem_coder
        self.sem_checker = sem_checker
        self.sem_renderer = sem_renderer
        self.status = status
        self.osd = osd
        self.send_image = send_image
        self.checker_mode = checker_mode
        self.ensemble_temperature = ensemble_temperature
        self.ref_vec = None
        self._coder_tasks: dict[int, asyncio.Task] = {}

    def start_coders(self, k_count: int) -> dict[int, asyncio.Task]:
        """Launch one coder per candidate at once (sem_coder caps concurrency)."""
        self._coder_tasks = {
            k: asyncio.create_task(self._generate(k)) for k in range(k_count)
        }
        return self._coder_tasks

    async def embed_reference(self) -> None:
        """Embed the reference image once, shared by every candidate's S2BV npz.

        Runs while the already-scheduled coders generate, so it adds no latency.
        """
        if self.embedder is not None and self.judge is not None and self.task.image_bytes:
            self.ref_vec = await self.embedder.embed_reference(self.task.image_bytes)

    async def leaf(self, k: int) -> Candidate | None:
        """Resolve bracket position k to a live candidate, or None for a walkover."""
        cand = await self._coder_tasks[k]
        digest = _code_digest(cand.js_code)
        if digest is None:
            return None  # coder failed -> walkover (drop_reason already set)

        # The dedup scan awaits every earlier coder, so candidate k is gated by the
        # slowest of 0..k-1 — the higher k, the longer the stall. Validation
        # (checker -> render -> judge views) does not depend on the scan result, so
        # it starts right away and overlaps it. The scan still covers all j < k, so
        # the outcome stays deterministic under a fixed seed; the only cost is
        # rendering a candidate that later turns out to be a duplicate.
        validate_task = asyncio.create_task(self._validate(cand))
        try:
            duplicate_of = None
            for j in range(k):
                if _code_digest((await self._coder_tasks[j]).js_code) == digest:
                    duplicate_of = j
                    break
        finally:
            await validate_task  # never leave the validation task dangling

        if duplicate_of is not None:
            cand.drop_reason = f"duplicate_of_k{duplicate_of}"
            return None  # the leader k=duplicate_of carries this code
        return cand if _is_live(cand) else None

    async def _generate(self, k: int) -> Candidate:
        task = self.task
        cand = Candidate(k=k, seed=task.seed + self.seed_offset + k)
        t0 = time.time()
        try:
            async with stage_guard(task, f"coder#k{k}", self.sem_coder, self.status):
                cand.js_code = await self.coder.code(
                    task_id=task.stem,
                    osd=self.osd,
                    image_bytes=task.image_bytes if self.send_image else None,
                    image_mime=task.image_mime,
                    actor_override=f"coder#k{k}",
                    seed_override=cand.seed,
                    temperature_override=self.ensemble_temperature,
                )
        except StageError as exc:
            cand.drop_reason = f"coder:{type(exc.cause).__name__}"
        except Exception as exc:
            cand.drop_reason = f"coder:{type(exc).__name__}"
        cand.elapsed_s = time.time() - t0
        return cand

    async def _validate(self, cand: Candidate) -> None:
        if cand.drop_reason is not None or cand.js_code is None:
            return
        task = self.task
        shadow = PipelineTask(stem=f"{task.stem}#k{cand.k}", image_url=task.image_url)
        shadow.js_code = cand.js_code
        shadow.image_bytes = task.image_bytes
        shadow.image_mime = task.image_mime
        t0 = time.time()
        async with self.sem_checker:
            try:
                await self.js_checker.process(shadow, mode=self.checker_mode)
            except Exception as exc:
                cand.drop_reason = f"checker:{type(exc).__name__}"
                cand.elapsed_s += time.time() - t0
                return
        cand.js_valid = shadow.js_valid
        cand.js_errors = list(shadow.js_errors or [])
        cand.scene_json = shadow.scene_json
        if not shadow.js_valid:
            cand.drop_reason = "checker"
            cand.elapsed_s += time.time() - t0
            return
        judge_views_wanted = (
            self.judge is not None
            and getattr(self.renderer, "config", None) is not None
            and getattr(self.renderer.config, "judge_multiview", False)
            and hasattr(self.renderer, "process_with_judge_views")
        )
        white: dict = {}
        gray: dict = {}
        async with self.sem_renderer:
            try:
                if judge_views_wanted:
                    white, gray = await self.renderer.process_with_judge_views(shadow)
                else:
                    await self.renderer.process(shadow)
            except Exception as exc:
                cand.drop_reason = f"renderer:{type(exc).__name__}"
                cand.elapsed_s += time.time() - t0
                return
        cand.rendered_png = shadow.rendered_png
        cand.render_errors = list(shadow.render_errors or [])
        if not shadow.rendered_png:
            cand.drop_reason = "renderer"
        cand.elapsed_s += time.time() - t0

        if cand.drop_reason is None and white:
            cand.judge_white_views, cand.judge_gray_views = white, gray
            if self.embedder is not None and self.ref_vec is not None:
                cand.judge_embeddings = await self.embedder.build_candidate_npz(
                    self.ref_vec, cand.judge_white_views
                )


async def multigen_first_iter(
    task: PipelineTask,
    *,
    coder,
    judge,
    embedder=None,
    js_checker,
    renderer,
    session_store,
    sem_coder: asyncio.Semaphore,
    sem_checker: asyncio.Semaphore,
    sem_renderer: asyncio.Semaphore,
    sem_judge: asyncio.Semaphore,
    coder_multimodal: bool,
    status: dict[str, str],
    ensemble_size: int,
    ensemble_temperature: float,
    render_from_object: bool = False,
    seed_offset: int = 0,
) -> None:
    """K-of-N generation + judge bracket. Replaces code_and_check + renderer
    on iteration 0 when `coder.ensemble_size > 1`.

    Mutates `task` so the rest of the iteration loop (critic → code_critic_repair)
    can continue on the winner.
    """

    osd = OSD.model_validate_json(task.osd) if task.osd is not None else None
    send_image = (osd is None) or coder_multimodal
    K = ensemble_size

    logger.info(
        f"[MULTIGEN] {task.stem} K={K} | temperature={ensemble_temperature} | "
        f"multimodal={send_image} | osd={'yes' if osd else 'no'}"
    )

    factory = _CandidateFactory(
        task,
        coder=coder,
        judge=judge,
        embedder=embedder,
        js_checker=js_checker,
        renderer=renderer,
        sem_coder=sem_coder,
        sem_checker=sem_checker,
        sem_renderer=sem_renderer,
        status=status,
        osd=osd,
        send_image=send_image,
        checker_mode="with_object" if render_from_object else "sanity",
        ensemble_temperature=ensemble_temperature,
        seed_offset=seed_offset,
    )

    # Launch every coder at once, then embed the reference while they run; each
    # candidate then flows coder -> dedup -> render -> bracket on its own, with no
    # barrier, so rendering and judging overlap the slower coders still in flight.
    coder_tasks = factory.start_coders(K)
    await factory.embed_reference()

    leaf_tasks: dict[int, asyncio.Task] = {
        k: asyncio.create_task(factory.leaf(k)) for k in range(K)
    }

    if judge is None or K <= 1:
        leaves = await asyncio.gather(*leaf_tasks.values())
        winner = next((c for c in leaves if _is_live(c)), None)
    else:
        winner = await _resolve_bracket(
            task=task, leaf_tasks=leaf_tasks, judge=judge, sem_judge=sem_judge,
        )

    # Every leaf (hence every coder) has resolved by the time the bracket returns.
    candidates = [coder_tasks[k].result() for k in range(K)]
    task.candidates = list(candidates)

    if not _is_live(winner):
        drops = [c.drop_reason for c in candidates]
        logger.warning(
            f"[MULTIGEN] {task.stem} K={K} | drops={drops} | ALL CANDIDATES FAILED"
        )
        raise StageError("multigen", RuntimeError("all candidates failed"))

    survivors = [c for c in candidates if _is_live(c)]
    unique = len({d for d in (_code_digest(c.js_code) for c in candidates) if d})
    # Rebuild deterministically by k (leaves resolve out of order now).
    task.multigen_pngs = [c.rendered_png for c in survivors]
    logger.info(
        f"[MULTIGEN] {task.stem} K={K} | unique={unique} | "
        f"survivors={[c.k for c in survivors]} | drops={[c.drop_reason for c in candidates]}"
    )

    _promote_winner(task, winner, candidates, session_store)
    logger.info(
        f"[MULTIGEN] {task.stem} winner=k{winner.k} | "
        f"js_bytes={len(winner.js_code.encode('utf-8')) if winner.js_code else 0}"
    )
