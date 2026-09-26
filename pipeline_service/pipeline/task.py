from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    """One multigen candidate produced by the coder in iteration 0."""

    k: int
    seed: int
    js_code: str | None = None
    js_valid: bool | None = None
    js_errors: list[str] = field(default_factory=list)
    scene_json: dict | None = None
    rendered_png: bytes | None = None
    render_errors: list[str] = field(default_factory=list)
    # Per-angle views for the multi-stage judge (name -> PNG bytes). Empty when
    # multiview rendering is disabled — the judge then runs grid-only stages.
    judge_white_views: dict[str, bytes] = field(default_factory=dict)
    judge_gray_views: dict[str, bytes] = field(default_factory=dict)
    # DINOv3 embeddings npz bytes (prompt + view_<name>...) for judge stage S2BV.
    judge_embeddings: bytes | None = None
    judge_views: Any = field(default=None, repr=False, compare=False)
    elapsed_s: float = 0.0
    drop_reason: str | None = None


@dataclass
class PipelineTask:
    """Single envelope threaded through every pipeline stage."""

    # Input
    stem: str
    image_url: str
    seed: int = 42

    # Fetched image
    image_bytes: bytes | None = None
    image_mime: str = "image/jpeg"
    judge_ref_url: str | None = field(default=None, repr=False)

    # Planner → OSD
    osd: str | None = None                 # JSON string for debug endpoints

    # Coder → JS
    js_code: str | None = None

    # JS Checker (mutated by JSCheckerModule.process)
    js_valid: bool | None = None
    js_errors: list[str] = field(default_factory=list)
    js_metrics: dict | None = None
    scene_json: dict | None = None
    js_stages_run: list[str] = field(default_factory=list)
    js_module_load_ms: float | None = None
    js_execution_ms: float | None = None
    js_total_ms: float | None = None

    # Renderer (mutated by RendererModule.process)
    multigen_pngs: list[bytes] = field(default_factory=list)  # rendered PNGs for each coder in the multigen ensemble
    rendered_png: bytes | None = None              # 2x2 grid
    render_ms: float | None = None
    render_errors: list[str] = field(default_factory=list)
    refinement_rendered_pngs: list[bytes] = field(default_factory=list)  # 2x2 grids from each refinement iteration

    # Refinement state (orchestrator)
    iteration: int = 0
    score_history: list[float] = field(default_factory=list)
    best_score: float = -1.0
    best_iter: int = -1
    best_js_code: str | None = None
    best_rendered_png: bytes | None = None

    # Multigen (orchestrator) — populated only when coder.ensemble_size > 1
    candidates: list[Candidate] = field(default_factory=list)
    winner_k: int | None = None

    # Lifecycle (orchestrator-private — not part of the public artifact)
    started_at: float = 0.0
    terminal: bool = False
    deadline_task: asyncio.Task | None = field(default=None, repr=False, compare=False)

    # Status
    failed: bool = False
    failure_reason: str | None = None
    failure_stage: str | None = None
    attempt: int = 0

    meta: dict[str, Any] = field(default_factory=dict)

    # Monitoring
    oom_retries: int = 0

