from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pipeline.task import PipelineTask


class ServiceState(Enum):
    STARTING = "starting"
    RUNNING  = "running"
    DOWN     = "down"
    

class MinerStatus(str, Enum):
    WARMING_UP = "warming_up"
    READY = "ready"
    GENERATING = "generating"
    COMPLETE = "complete"
    REPLACE = "replace"


class MinerState:
    """Per-batch state."""

    def __init__(self) -> None:
        self.status: MinerStatus = MinerStatus.WARMING_UP
        self.total: int = 0
        self.batch_stems: list[str] = []
        self.seed: int = 0
        self.tasks: dict[str, "PipelineTask"] = {}
        # Per-client probe state, keyed by llm_clients name (e.g. "vision",
        # "coder"). Populated by _watch_models from app.state.checkers. Empty
        # in cloud-only deployments — see models_ready() below.
        self.llm_status: dict[str, ServiceState] = {}
        self.replacements_remaining: int = 0
        # Host diagnostics: preflight metrics, coder-probe tok/s, batch timing. Rendered into the
        # `// miner-diag:` header of every module in /results (serve.py).
        self.diag: dict[str, Any] = {}
        self.batch_started_at: float | None = None
        self.batch_index: int = 0

    def reset_for_batch(self, stems: list[str], seed: int) -> None:
        self.total = len(stems)
        self.tasks = {}
        self.batch_stems = stems
        self.seed = seed
        self.status = MinerStatus.GENERATING
        self.batch_started_at = time.time()
        self.batch_index += 1

    def record_task(self, task: "PipelineTask") -> None:
        self.tasks[task.stem] = task

    @property
    def progress(self) -> int:
        return len(self.tasks)

    @property
    def results(self) -> dict[str, str]:
        """Successful tasks keyed by stem, value = js_code."""
        return {
            stem: t.js_code
            for stem, t in self.tasks.items()
            if not t.failed and t.js_code
        }

    @property
    def failed(self) -> dict[str, str]:
        """Failed tasks keyed by stem, value = failure reason."""
        out: dict[str, str] = {}
        for stem, t in self.tasks.items():
            if t.failed or not t.js_code:
                out[stem] = t.failure_reason or "no code generated"
        return out

    def mark_complete(self) -> None:
        self.status = MinerStatus.COMPLETE
    
    def set_llm_status(self, name: str, st: ServiceState) -> None:
        self.llm_status[name] = st

    def models_ready(self) -> bool:
        if not self.llm_status:
            return True
        return all(s == ServiceState.RUNNING for s in self.llm_status.values())
    
    def to_status_response(self) -> dict[str, Any]:
        active = self.status in (MinerStatus.GENERATING, MinerStatus.COMPLETE)
        return {
            "status": self.status.value,
            "progress": self.progress if active else 0,
            "total":    self.total if active else 0,
            "payload":  None,
        }
