from __future__ import annotations

import asyncio
import base64
import time

from openai import AsyncOpenAI

from config.settings import ActorConfig
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
    """Pairwise visual judge backed by the duel-eval multi-stage pipeline."""
    
    actor = "judge"

    def __init__(
        self,
        client: AsyncOpenAI,
        settings: ActorConfig,
        *,
        max_stage: int = 4,
        s1_concurrency: int = 8,
    ) -> None:
        super().__init__(client, settings)
        self.max_stage = max_stage
        self.s1_concurrency = s1_concurrency
        self.reasoning_effort = settings.reasoning_effort
        if not getattr(settings, "explain", False):
            # The explain call is opponent-independent and only fills detail["issues"]; skipping it saves the heaviest
            # image request per duel (reference + both grids) without changing any verdict.
            async def _no_explain(*_a, **_k):
                return _ms._neutral_issues().issues
            _ms._explain_run = _no_explain
            logger.info("[Judge] explain call disabled (actors.judge.explain=false)")

    async def _draw_tiebreak(
        self, left_views: ViewsAdapter, right_views: ViewsAdapter
    ) -> tuple[str, str, bool]:
        """Break a duel draw by DINOv3 best-view similarity to the reference."""
        try:
            emb_a, emb_b = await asyncio.gather(
                left_views.fetch_embeddings(), right_views.fetch_embeddings()
            )
            sim_a = best_view_similarity(emb_a)
            sim_b = best_view_similarity(emb_b)
        except Exception as exc:  
            logger.warning(f"[Judge] draw tie-break failed: {exc!r}")
            sim_a = sim_b = None

        if sim_a is not None and sim_b is not None and sim_a != sim_b:
            side = "A" if sim_a > sim_b else "B"
            return side, f"DINO tie-break (simA={sim_a:.3f} simB={sim_b:.3f} -> {side})", True
        return "A", "draw -> A (no DINO signal)", False

    async def compare(
        self,
        *,
        task_id: str,
        match_label: str,
        reference_bytes: bytes,
        reference_mime: str,
        render_a: bytes,
        render_b: bytes,
        white_views_a: dict[str, bytes] | None = None,
        white_views_b: dict[str, bytes] | None = None,
        gray_views_a: dict[str, bytes] | None = None,
        gray_views_b: dict[str, bytes] | None = None,
        embeddings_a: bytes | None = None,
        embeddings_b: bytes | None = None,
    ) -> JudgeVerdict:
        _t0 = time.monotonic()
        prompt_url = _data_url(reference_bytes, reference_mime)
        grid_a = _b64(render_a)
        grid_b = _b64(render_b)

        left_views = ViewsAdapter(
            white_views=_views_to_b64(white_views_a),
            gray_views=_views_to_b64(gray_views_a),
            grid=grid_a,
            embeddings=_b64(embeddings_a) if embeddings_a else None,
        )
        right_views = ViewsAdapter(
            white_views=_views_to_b64(white_views_b),
            gray_views=_views_to_b64(gray_views_b),
            grid=grid_b,
            embeddings=_b64(embeddings_b) if embeddings_b else None,
        )

        _enc_s = time.monotonic() - _t0
        _payload_kb = (len(prompt_url) + len(grid_a) + len(grid_b)
                       + sum(len(v) for v in left_views._white.values()) + sum(len(v) for v in right_views._white.values())
                       + sum(len(v) for v in left_views._gray.values()) + sum(len(v) for v in right_views._gray.values())) / 1024
        prefix = f"[Judge {match_label}]"
        logger.info(f"{prefix} [JUDGE_TIMING] b64_encode={_enc_s*1000:.0f}ms payload={_payload_kb:.0f}KB")
        logger.info(
            f"{prefix} Started Task {task_id} | Model: {self.model} | "
            f"max_stage={self.max_stage} | "
            f"Ref KB: {len(reference_bytes) / 1024:.1f} | "
            f"A KB: {len(render_a) / 1024:.1f} | B KB: {len(render_b) / 1024:.1f} | "
            f"white A/B: {len(white_views_a or {})}/{len(white_views_b or {})}"
        )

        sem = asyncio.Semaphore(self.s1_concurrency)
        winner, detail = await evaluate_duel(
            self.client,
            sem,
            prompt_url,
            left_views,
            right_views,
            seed=self.seed,
            model=self.model,
            log_id=f"{task_id} {match_label}",
            max_stage=self.max_stage,
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
