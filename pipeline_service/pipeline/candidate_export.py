"""Export of every multigen candidate (code, renders, judge inputs) for offline dry-run experiments.

One directory (or zip) per task attempt:
    manifest.json             task fields, per-candidate metadata, task.meta (judge duel records, similarities)
    reference.<ext>           prompt image exactly as downloaded
    k07.js                    candidate code (every candidate whose coder returned code, dropped ones included)
    renders/k07/grid.png      2x2 grid; white_<view>.png, gray_<view>.png; embeddings.npz (DINOv3, judge S2BV input)
The render files are the judge's inputs, so any pair can be re-judged offline.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel

from pipeline.task import PipelineTask

_MIME_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
_STORED_SUFFIXES = (".png", ".jpg", ".webp", ".npz")


class CandidateRecord(BaseModel):
    k: int
    seed: int
    drop_reason: str | None
    js_valid: bool | None
    js_errors: list[str]
    render_errors: list[str]
    elapsed_s: float
    js_file: str | None
    render_dir: str | None
    white_views: list[str]
    gray_views: list[str]
    has_embeddings: bool


class CandidateManifest(BaseModel):
    stem: str
    image_url: str
    reference_file: str | None
    seed: int
    attempt: int
    batch_index: int
    winner_k: int | None
    failed: bool
    failure_reason: str | None
    candidates: list[CandidateRecord]
    meta: dict


def _tag(k: int, width: int) -> str:
    return f"k{k:0{width}d}"


def build_manifest(task: PipelineTask, batch_index: int) -> CandidateManifest:
    width = max(2, len(str(max((c.k for c in task.candidates), default=0))))
    records = [
        CandidateRecord(
            k=c.k, seed=c.seed, drop_reason=c.drop_reason, js_valid=c.js_valid,
            js_errors=list(c.js_errors), render_errors=list(c.render_errors), elapsed_s=round(c.elapsed_s, 2),
            js_file=f"{_tag(c.k, width)}.js" if c.js_code else None,
            render_dir=f"renders/{_tag(c.k, width)}" if c.rendered_png else None,
            white_views=sorted(c.judge_white_views), gray_views=sorted(c.judge_gray_views),
            has_embeddings=c.judge_embeddings is not None,
        )
        for c in task.candidates
    ]
    ref_ext = _MIME_EXT.get(task.image_mime, "bin")
    return CandidateManifest(
        stem=task.stem, image_url=task.image_url,
        reference_file=f"reference.{ref_ext}" if task.image_bytes else None,
        seed=task.seed, attempt=task.attempt, batch_index=batch_index, winner_k=task.winner_k,
        failed=task.failed, failure_reason=task.failure_reason, candidates=records, meta=task.meta,
    )


def iter_files(task: PipelineTask, batch_index: int, *, renders: bool) -> Iterator[tuple[str, bytes]]:
    """(relative path, bytes) for every file of the export; renders=False leaves out PNGs and embeddings."""
    manifest = build_manifest(task, batch_index)
    yield "manifest.json", manifest.model_dump_json(indent=1).encode()
    if manifest.reference_file and task.image_bytes:
        yield manifest.reference_file, task.image_bytes
    for cand, rec in zip(task.candidates, manifest.candidates):
        if rec.js_file and cand.js_code:
            yield rec.js_file, cand.js_code.encode("utf-8")
        if not renders or rec.render_dir is None or not cand.rendered_png:
            continue
        yield f"{rec.render_dir}/grid.png", cand.rendered_png
        for name, png in cand.judge_white_views.items():
            yield f"{rec.render_dir}/white_{name}.png", png
        for name, png in cand.judge_gray_views.items():
            yield f"{rec.render_dir}/gray_{name}.png", png
        if cand.judge_embeddings:
            yield f"{rec.render_dir}/embeddings.npz", cand.judge_embeddings


def build_zip(task: PipelineTask, batch_index: int, *, renders: bool) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, data in iter_files(task, batch_index, renders=renders):
            kind = zipfile.ZIP_STORED if path.endswith(_STORED_SUFFIXES) else zipfile.ZIP_DEFLATED
            zf.writestr(path, data, compress_type=kind)
    return buf.getvalue()


def dump_dir(root: Path, task: PipelineTask, batch_index: int) -> Path:
    suffix = f"_attempt{task.attempt}" if task.attempt else ""
    return root / f"batch_{batch_index:03d}" / f"{task.stem}{suffix}"


def write_dump(task: PipelineTask, root: Path, batch_index: int) -> tuple[Path, int]:
    """Write the full export (renders included) under root; returns (directory, file count). Blocking file IO."""
    out = dump_dir(root, task, batch_index)
    count = 0
    for path, data in iter_files(task, batch_index, renders=True):
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        count += 1
    return out, count
