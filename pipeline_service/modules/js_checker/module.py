from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
from pathlib import Path

from logger_config import logger
from modules.base import BaseModule
from modules.js_checker.settings import JSCheckerConfig
from pipeline.task import PipelineTask

_RUNNER_JS = Path(__file__).parent / "validate_runner.mjs"
_SERIALIZE_RUNNER_JS = Path(__file__).parent / "serialize_runner.mjs"
_ROTATE_RUNNER_JS = Path(__file__).parent / "rotate_runner.mjs"

_RUNNERS = {
    "sanity": _RUNNER_JS,
    "with_object": _SERIALIZE_RUNNER_JS,
}


def _node_cwd(runner: Path) -> str:
    """First directory that actually carries node_modules for the runner's imports."""
    preferred = os.environ.get("NODE_CWD", str(runner.parent))
    for candidate in (preferred, "/workspace", str(runner.parent)):
        if os.path.isdir(os.path.join(candidate, "node_modules")):
            return candidate
    return preferred


class JSCheckerModule(BaseModule):

    def __init__(self, config: JSCheckerConfig) -> None:
        self.config = config

    async def startup(self) -> None:
        for runner in _RUNNERS.values():
            if not runner.exists():
                logger.warning(f"js_checker runner not found at {runner}")

    @staticmethod
    def _nan_bbox_detail(metrics) -> str:
        """Return a failure detail if the validator's bbox carries null/NaN/inf coordinates, else ''."""
        if not isinstance(metrics, dict):
            return ""
        bbox = metrics.get("bbox")
        if not isinstance(bbox, dict):
            return ""
        bad = []
        for side in ("min", "max"):
            coords = bbox.get(side)
            if not isinstance(coords, dict):
                bad.append(f"{side}=missing")
                continue
            for axis in ("x", "y", "z"):
                v = coords.get(axis)
                if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
                    bad.append(f"{side}.{axis}={v!r}")
        if not bad:
            return ""
        return f"bbox has non-finite coordinates ({', '.join(bad[:6])}); vertices={metrics.get('vertices', '?')}"

    async def process(self, task: PipelineTask, mode: str = "sanity") -> PipelineTask:
        logger.info(
            f"[JS_CHECK] '{task.stem}' start | mode={mode} | "
            f"js_code={len(task.js_code) if task.js_code else 0} bytes"
        )

        if not task.js_code:
            task.failed = True
            task.failure_reason = "No JS code to validate"
            logger.warning(f"[JS_CHECK] '{task.stem}' skip — no JS code")
            return task

        result = await self._validate(task.js_code, mode)

        task.js_valid = result.get("passed", False)
        task.js_stages_run = result.get("stagesRun", [])
        task.js_metrics = result.get("metrics")
        task.js_module_load_ms = result.get("moduleLoadMs")
        task.js_execution_ms = result.get("executionMs")
        task.js_total_ms = result.get("totalMs")

        if mode == "with_object" and task.js_valid:
            task.scene_json = result.get("object")

        failures = list(result.get("failures", []))
        if task.js_valid:
            nan_detail = self._nan_bbox_detail(task.js_metrics)
            if nan_detail:
                task.js_valid = False
                failures.append({"rule": "NAN_GEOMETRY", "detail": nan_detail})
        task.js_errors = []
        for f in failures:
            rule = f.get("rule", "UNKNOWN")
            detail = f.get("detail", "")
            task.js_errors.append(f"{rule}: {detail}" if detail else rule)

        if not task.js_valid:
            task.failed = True
            task.failure_reason = f"JS validation failed: {'; '.join(task.js_errors[:3])}"
            logger.warning(
                f"[JS_CHECK] '{task.stem}' FAIL | "
                f"stages={task.js_stages_run} | "
                f"errors={task.js_errors}"
            )
        else:
            m = task.js_metrics or {}
            bbox = m.get("bbox") or {}
            bbox_str = ""
            if bbox:
                mn = bbox.get("min") or {}
                mx = bbox.get("max") or {}

                def _f(d: dict, k: str) -> float:
                    # Tolerate missing keys and explicit `None` in values.
                    v = d.get(k)
                    try:
                        return float(v) if v is not None else 0.0
                    except (TypeError, ValueError):
                        return 0.0

                bbox_str = (
                    f"[{_f(mn,'x'):.2f},{_f(mn,'y'):.2f},{_f(mn,'z'):.2f}]→"
                    f"[{_f(mx,'x'):.2f},{_f(mx,'y'):.2f},{_f(mx,'z'):.2f}]"
                )
            load_ms = task.js_module_load_ms or 0
            exec_ms = task.js_execution_ms or 0
            total_ms = task.js_total_ms or 0
            logger.info(
                f"[JS_CHECK] '{task.stem}' PASS | "
                f"vertices={m.get('vertices', '?')} drawCalls={m.get('drawCalls', '?')} "
                f"depth={m.get('maxDepth', '?')} instances={m.get('instances', '?')} "
                f"texBytes={m.get('textureBytes', '?')} | "
                f"bbox={bbox_str} | "
                f"timing: load={load_ms/1000:.1f}s exec={exec_ms/1000:.1f}s total={total_ms/1000:.1f}s"
            )

        return task

    async def rotate_source(self, code: str, steps: list[tuple[str, str]]) -> str | None:
        """Write `steps` (axis, angle expression) into the module's own generate(), before every return.

        Returns the rewritten source, or None when the module has no shape the injector can edit —
        the caller then keeps the original candidate.
        """
        if not code or not steps:
            return None
        if not _ROTATE_RUNNER_JS.exists():
            logger.warning(f"[JS_CHECK] rotate runner missing at {_ROTATE_RUNNER_JS}")
            return None

        tmp_dir = tempfile.mkdtemp(prefix="jsrotate_")
        try:
            code_path = os.path.join(tmp_dir, "module.mjs")
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)
            args = [arg for axis, angle in steps for arg in (axis, angle)]
            proc = await asyncio.create_subprocess_exec(
                self.config.node_binary,
                str(_ROTATE_RUNNER_JS), code_path, *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=_node_cwd(_ROTATE_RUNNER_JS),
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.warning("[JS_CHECK] rotate runner timed out")
                return None
            if proc.returncode != 0:
                logger.warning(
                    f"[JS_CHECK] rotate runner failed: {stderr.decode('utf-8', errors='replace').strip()[:200]}"
                )
                return None
            rotated = stdout.decode("utf-8", errors="replace")
            return rotated or None
        except Exception as exc:  # noqa: BLE001 - rotation is optional, never fails the candidate
            logger.warning(f"[JS_CHECK] rotate runner error: {exc}")
            return None
        finally:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _validate(self, code: str, mode: str = "sanity") -> dict:
        runner = _RUNNERS.get(mode)
        if runner is None:
            return {"passed": False, "failures": [{"rule": "BAD_MODE", "detail": mode}]}
        if not runner.exists():
            return {"passed": False, "failures": [{"rule": "RUNNER_MISSING", "detail": str(runner)}]}

        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="jschecker_")
            code_path = os.path.join(tmp_dir, "module.mjs")
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)

            node_cwd = _node_cwd(runner)

            proc = await asyncio.create_subprocess_exec(
                self.config.node_binary,
                str(runner), code_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=node_cwd,
            )

            outer_timeout = self.config.execution_timeout_ms / 1000.0 + 8.0
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=outer_timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return {"passed": False, "failures": [{"rule": "TIMEOUT_EXCEEDED", "detail": "outer Python timeout"}]}

            if proc.returncode != 0:
                err_text = stderr.decode("utf-8", errors="replace").strip()
                return {"passed": False, "failures": [{"rule": "EXECUTION_THREW", "detail": err_text[:300]}]}

            result_text = stdout.decode("utf-8", errors="replace").strip()
            if not result_text:
                return {"passed": False, "failures": [{"rule": "EXECUTION_THREW", "detail": "empty runner output"}]}

            return json.loads(result_text)

        except json.JSONDecodeError:
            return {"passed": False, "failures": [{"rule": "EXECUTION_THREW", "detail": "invalid runner JSON"}]}
        except Exception as exc:
            logger.warning(f"Validator execution error: {exc}")
            return {"passed": False, "failures": [{"rule": "EXECUTION_THREW", "detail": str(exc)[:200]}]}
        finally:
            if tmp_dir:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
