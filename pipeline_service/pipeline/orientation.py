"""Bring a candidate to the orientation the judge scores it in.

The judge's four S1 angles are all front-family, so a module built sideways or lying flat loses its
comparisons whatever its geometry is worth.
A cheap low-res probe names the side that matches the reference; the rotation is then written into the
candidate's own generate(), so the submitted code is what produces the oriented object.
"""
from __future__ import annotations

import contextlib
import io
from enum import Enum

from PIL import Image
from pydantic import BaseModel

from config.settings import OrientationConfig
from logger_config import logger
from pipeline.task import PipelineTask


class ProbeView(str, Enum):
    """Renderer presets the probe renders (render_service/render_runner.mjs VIEW_PRESETS)."""

    FRONT = "front"
    RIGHT = "right"
    BACK = "back"
    LEFT = "left"
    TOP_DOWN = "top_down"


class RotationAxis(str, Enum):
    X = "x"
    Y = "y"


class RotationStep(BaseModel):
    axis: RotationAxis
    angle_expr: str


class Rotation(BaseModel):
    best_view: ProbeView
    steps: list[RotationStep]
    image_rotation_deg: int | None = None


class OrientationResult(BaseModel):
    best_view: ProbeView | None = None
    sims: dict[str, float] = {}
    rotation: Rotation | None = None
    applied: bool = False
    reason: str | None = None


_SIDE_YAW: dict[ProbeView, str] = {
    ProbeView.RIGHT: "-Math.PI / 2",
    ProbeView.BACK: "Math.PI",
    ProbeView.LEFT: "Math.PI / 2",
}
_TIP_STEP = RotationStep(axis=RotationAxis.X, angle_expr="Math.PI / 2")
_TOP_DOWN_YAW: dict[int, str] = {
    0: "0",
    90: "Math.PI / 2",
    180: "Math.PI",
    270: "-Math.PI / 2",
}
_IMAGE_ROTATIONS = (90, 180, 270)


def _rotated_png(png: bytes, degrees: int) -> bytes:
    with Image.open(io.BytesIO(png)) as img:
        out = io.BytesIO()
        img.rotate(degrees, expand=True).save(out, format="PNG")
        return out.getvalue()


def _top_down_variants(png: bytes) -> dict[str, bytes]:
    """The top-down render plus its in-plane rotations, keyed for the embedder."""
    variants = {ProbeView.TOP_DOWN.value: png}
    for deg in _IMAGE_ROTATIONS:
        try:
            variants[f"{ProbeView.TOP_DOWN.value}_r{deg}"] = _rotated_png(png, deg)
        except Exception as exc:  # noqa: BLE001 - a missing variant only narrows the yaw choice
            logger.warning(f"[ORIENT] top-down rotation {deg} failed: {exc}")
    return variants


async def _decide(embedder, ref_vec, views: dict[str, bytes]) -> tuple[dict[str, float], Rotation | None]:
    """Best probe view -> the rotation that faces it front, or None when it already does."""
    vecs = await embedder.embed_views(views)
    if not vecs:
        return {}, None
    sims = embedder.similarities(ref_vec, vecs)
    if not sims:
        return {}, None
    best = ProbeView(max(sims, key=sims.get))

    if best is ProbeView.FRONT:
        return sims, None
    if best in _SIDE_YAW:
        return sims, Rotation(
            best_view=best,
            steps=[RotationStep(axis=RotationAxis.Y, angle_expr=_SIDE_YAW[best])],
        )

    top_png = views.get(ProbeView.TOP_DOWN.value)
    steps = [_TIP_STEP]
    image_deg = 0
    if top_png:
        variants = _top_down_variants(top_png)
        var_vecs = await embedder.embed_views(variants)
        var_sims = embedder.similarities(ref_vec, var_vecs or {})
        if var_sims:
            sims.update(var_sims)
            best_variant = max(var_sims, key=var_sims.get)
            image_deg = int(best_variant.rsplit("_r", 1)[1]) if "_r" in best_variant else 0
            yaw = _TOP_DOWN_YAW.get(image_deg, "0")
            if yaw != "0":
                steps = [RotationStep(axis=RotationAxis.Y, angle_expr=yaw), _TIP_STEP]
    return sims, Rotation(best_view=best, steps=steps, image_rotation_deg=image_deg)


async def orient_task(
    task: PipelineTask,
    *,
    renderer,
    embedder,
    ref_vec,
    js_checker,
    checker_mode: str,
    config: OrientationConfig,
    sem_render=None,
    sem_check=None,
) -> OrientationResult:
    """Probe, decide, and rewrite `task.js_code` / `task.scene_json` in place when a rotation helps.

    The probe render and the re-check hold their own stage's semaphore, so neither waits on the other's
    pool. Every failure path leaves the task untouched: a candidate is never dropped for failing to rotate.
    """
    if not config.enabled or embedder is None or ref_vec is None:
        return OrientationResult(reason="disabled")

    render_guard = sem_render if sem_render is not None else contextlib.nullcontext()
    check_guard = sem_check if sem_check is not None else contextlib.nullcontext()

    async with render_guard:
        views = await renderer.render_views(
            task, [v.value for v in ProbeView], img_size=config.img_size
        )
    if not views:
        return OrientationResult(reason="probe_failed")

    try:
        sims, rotation = await _decide(embedder, ref_vec, views)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[ORIENT] '{task.stem}' decision failed: {type(exc).__name__}: {exc}")
        return OrientationResult(reason="decision_failed")

    if rotation is None:
        best = ProbeView(max(sims, key=sims.get)) if sims else None
        logger.info(f"[ORIENT] '{task.stem}' best={best.value if best else '?'} | no rotation")
        return OrientationResult(best_view=best, sims=sims, reason="already_front")

    steps = [(s.axis.value, s.angle_expr) for s in rotation.steps]
    rotated = await js_checker.rotate_source(task.js_code, steps)
    if not rotated:
        logger.warning(f"[ORIENT] '{task.stem}' best={rotation.best_view.value} | injection failed")
        return OrientationResult(
            best_view=rotation.best_view, sims=sims, rotation=rotation, reason="injection_failed"
        )

    probe = PipelineTask(stem=f"{task.stem}~rot", image_url=task.image_url)
    probe.js_code = rotated
    async with check_guard:
        await js_checker.process(probe, mode=checker_mode)
    if not probe.js_valid:
        logger.warning(
            f"[ORIENT] '{task.stem}' best={rotation.best_view.value} | rotated code rejected: "
            f"{probe.js_errors[:2]}"
        )
        return OrientationResult(
            best_view=rotation.best_view, sims=sims, rotation=rotation, reason="checker_rejected"
        )

    task.js_code = rotated
    task.scene_json = probe.scene_json
    task.js_metrics = probe.js_metrics
    logger.info(
        f"[ORIENT] '{task.stem}' best={rotation.best_view.value} | "
        f"applied {' + '.join(f'{s.axis.value} {s.angle_expr}' for s in rotation.steps)}"
    )
    return OrientationResult(best_view=rotation.best_view, sims=sims, rotation=rotation, applied=True)
