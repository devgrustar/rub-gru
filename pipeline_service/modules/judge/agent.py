from __future__ import annotations

import asyncio
import base64

from openai import AsyncOpenAI

from config.settings import JudgeConfig
from logger_config import logger
from modules.base_agent import BaseAgent
from modules.judge import multi_stage as _ms
from modules.judge.multi_stage import ViewsAdapter, best_view_similarity, evaluate_duel
from modules.judge.schema import JudgeVerdict


def _b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode()


def _data_url(image_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{_b64(image_bytes)}"


def _views_to_b64(views: dict[str, bytes] | None) -> dict[str, str]:
    """Convert a name->PNG-bytes dict into the name->base64 dict ViewsAdapter wants."""
    if not views:
        return {}
    return {name: _b64(data) for name, data in views.items() if data}


def _build_views(
    grid: bytes,
    white: dict[str, bytes] | None,
    gray: dict[str, bytes] | None,
    embeddings: bytes | None,
) -> ViewsAdapter:
    return ViewsAdapter(
        white_views=_views_to_b64(white), gray_views=_views_to_b64(gray), grid=_b64(grid),
        embeddings=_b64(embeddings) if embeddings else None,
    )


def _payload_kb(views: ViewsAdapter) -> float:
    return (
        len(views._grid)
        + sum(len(v) for v in views._white.values())
        + sum(len(v) for v in views._gray.values())
    ) / 1024


_WINNER_TO_SIDE: dict[str, str] = {"left": "A", "right": "B", "draw": "A"}

_CONFIDENCE_BY_STAGE: list[tuple[str, float]] = [
    ("S1", 0.85),
    ("strong_voter", 0.85),
    ("S2 consensus", 0.75),
    ("S3", 0.65),
    ("S4 step-down", 0.70),
]


def _confidence_from_decided_by(decided_by: str, winner: str) -> float:
    if winner == "draw":
        return 0.5
    for marker, conf in _CONFIDENCE_BY_STAGE:
        if marker in decided_by:
            return conf
    return 0.6


class JudgeAgent(BaseAgent):
    """Pairwise visual judge backed by the multi-stage duel pipeline."""

    actor = "judge"

    def __init__(self, client: AsyncOpenAI, settings: JudgeConfig) -> None:
        super().__init__(client, settings)
        self.max_stage = settings.max_stage.value
        self.s1_concurrency = settings.s1_concurrency
        self.s1_early_stop = settings.s1_early_stop
        if not settings.explain:
            async def _no_explain(*_a, **_k):
                return _ms._neutral_issues().issues
            _ms._explain_run = _no_explain
            logger.info("[Judge] explain call disabled (actors.judge.explain=false)")

    async def encode_reference(self, image_bytes: bytes, mime: str) -> str:
        """Reference image as a data URL, encoded off the event loop."""
        return await asyncio.to_thread(_data_url, image_bytes, mime)

    async def encode_views(
        self,
        *,
        grid: bytes,
        white: dict[str, bytes] | None,
        gray: dict[str, bytes] | None,
        embeddings: bytes | None,
    ) -> ViewsAdapter:
        """One candidate's judge inputs, base64-encoded once for every duel it plays."""
        return await asyncio.to_thread(_build_views, grid, white, gray, embeddings)

    def early_stop_for_round(self, round_no: int) -> bool:
        return self.s1_early_stop.enabled and round_no <= self.s1_early_stop.max_round

    async def similarity(self, views: ViewsAdapter) -> float | None:
        """DINOv3 best-view cosine similarity to the reference; None without embeddings."""
        try:
            return best_view_similarity(await views.fetch_embeddings())
        except Exception as exc:
            logger.warning(f"[Judge] similarity failed: {exc!r}")
            return None

    async def _draw_tiebreak(
        self, left_views: ViewsAdapter, right_views: ViewsAdapter
    ) -> tuple[str, str, bool]:
        """Break a duel draw by DINOv3 best-view similarity to the reference."""
        sim_a, sim_b = await asyncio.gather(self.similarity(left_views), self.similarity(right_views))
        if sim_a is not None and sim_b is not None and sim_a != sim_b:
            side = "A" if sim_a > sim_b else "B"
            return side, f"DINO tie-break (simA={sim_a:.3f} simB={sim_b:.3f} -> {side})", True
        return "A", "draw -> A (no DINO signal)", False

    async def compare(
        self,
        *,
        task_id: str,
        match_label: str,
        prompt_url: str,
        left_views: ViewsAdapter,
        right_views: ViewsAdapter,
        s1_temperature: float = 0.0,
        seed: int | None = None,
        max_stage: int | None = None,
        early_stop: bool | None = None,
    ) -> JudgeVerdict:
        seed = self.seed if seed is None else seed
        max_stage = self.max_stage if max_stage is None else max_stage
        use_early_stop = self.s1_early_stop.enabled if early_stop is None else early_stop
        prefix = f"[Judge {match_label}]"
        logger.info(
            f"{prefix} Started Task {task_id} | Model: {self.model} | "
            f"max_stage={max_stage} | "
            f"payload A/B KB: {_payload_kb(left_views):.0f}/{_payload_kb(right_views):.0f} | "
            f"white A/B: {len(left_views._white)}/{len(right_views._white)}"
            + (f" | s1_temperature={s1_temperature}" if s1_temperature else "")
            + (" | s1_early_stop" if use_early_stop else "")
        )

        sem = asyncio.Semaphore(self.s1_concurrency)
        winner, detail = await evaluate_duel(
            self.client,
            sem,
            prompt_url,
            left_views,
            right_views,
            seed=seed,
            model=self.model,
            log_id=f"{task_id} {match_label}",
            max_stage=max_stage,
            s1_temperature=s1_temperature,
            early_stop=self.s1_early_stop if use_early_stop else None,
        )

        if winner == "draw":
            side, tie_note, had_signal = await self._draw_tiebreak(left_views, right_views)
            base = str(detail.get("decided_by", "")) or "draw"
            decided_by = f"{base} | {tie_note}"
            reason = str(detail.get("issues", "")) or tie_note
            confidence = 0.55 if had_signal else 0.5
        else:
            side = _WINNER_TO_SIDE[winner]  # left -> A, right -> B
            decided_by = str(detail.get("decided_by", ""))
            reason = str(detail.get("issues", "")) or f"decided by {decided_by or 'multi-stage'}"
            confidence = _confidence_from_decided_by(decided_by, winner)

        verdict = JudgeVerdict(
            winner=side,
            reason=reason,
            confidence=confidence,
            decided_by=decided_by,
            detail={
                k: detail[k]
                for k in ("s1_slim", "s2_slim", "s3_slim", "s4_slim", "decided_by")
                if k in detail
            },
        )
        logger.info(
            f"{prefix} Finished Task {task_id} | duel_winner={winner} -> {side} | "
            f"decided_by={decided_by} | confidence={confidence:.2f} | "
            f"reason={reason[:120]}"
        )
        return verdict
