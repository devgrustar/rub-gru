from __future__ import annotations

import asyncio
import base64
import os
import random
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from logger_config import logger
from modules.base import BaseModule
from modules.renderer.grid import GRID_KEYS, GRID_VIEWS, compose_grid
from modules.renderer.settings import RendererConfig
from pipeline.task import PipelineTask

_RUNNER_JS = Path(__file__).parent / "render_service" / "render_runner.mjs"

_RETRYABLE_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)


@dataclass
class _Sidecar:
    idx: int
    port: int
    proc: asyncio.subprocess.Process
    client: httpx.AsyncClient
    log_tasks: list[asyncio.Task] = field(default_factory=list)
    browser_pid: int | None = None


@dataclass
class _Slot:
    idx: int
    sidecar: _Sidecar | None = None
    task: asyncio.Task | None = None


class RendererModule(BaseModule):
    _PING_INTERVAL_S = 5.0
    _PING_TIMEOUT_S = 15.0
    _PING_STRIKES = 6
    _SPAWN_BACKOFF_INITIAL_S = 1.0
    _SPAWN_BACKOFF_MAX_S = 30.0
    _TERM_GRACE_S = 5.0
    _HEALTHY_UPTIME_RESET_S = 60.0

    def __init__(self, config: RendererConfig) -> None:
        self.config = config
        self._slots: list[_Slot] = []
        self._dispatch_counter = 0
        self._node_cwd: str | None = None
        self._shutting_down = False

    async def startup(self) -> None:
        if not _RUNNER_JS.exists():
            logger.warning(f"[RENDERER] runner missing at {_RUNNER_JS}, disabling")
            return

        self._node_cwd = self._resolve_node_cwd()
        count = max(1, self.config.sidecar_count)
        logger.info(
            f"[RENDERER] starting {count} sidecar(s) | node={self.config.node_binary} "
            f"runner={_RUNNER_JS} base_port={self.config.sidecar_port} cwd={self._node_cwd} "
            f"pool_size_per_sidecar={self.config.pool_size}"
        )

        self._slots = [_Slot(idx=i) for i in range(count)]
        for slot in self._slots:
            slot.task = asyncio.create_task(self._supervise(slot))

        deadline = time.monotonic() + self.config.startup_timeout_s + 30.0
        while time.monotonic() < deadline:
            ready = sum(1 for s in self._slots if s.sidecar is not None)
            if ready:
                logger.info(
                    f"[RENDERER] {ready}/{count} sidecar(s) ready "
                    f"(remaining come up in background), slots per sidecar={self.config.pool_size}"
                )
                return
            await asyncio.sleep(0.5)

        await self.shutdown()
        raise RuntimeError(
            f"[RENDERER] no sidecar became ready within {self.config.startup_timeout_s + 30.0:.0f}s"
        )

    async def _supervise(self, slot: _Slot) -> None:
        backoff = self._SPAWN_BACKOFF_INITIAL_S
        while not self._shutting_down:
            try:
                sc = await self._spawn_sidecar(slot.idx)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    f"[RENDERER] sidecar #{slot.idx} spawn failed: {exc} — retry in {backoff:.0f}s"
                )
                await asyncio.sleep(backoff + random.uniform(0.0, 0.5))
                backoff = min(backoff * 2.0, self._SPAWN_BACKOFF_MAX_S)
                continue

            ready_at = time.monotonic()
            slot.sidecar = sc
            try:
                await self._monitor(sc)
            finally:
                slot.sidecar = None
                await self._teardown_sidecar(sc)

            if time.monotonic() - ready_at > self._HEALTHY_UPTIME_RESET_S:
                backoff = self._SPAWN_BACKOFF_INITIAL_S

    async def _spawn_sidecar(self, idx: int) -> _Sidecar:
        port = self.config.sidecar_port + idx
        static_port = self.config.static_port_base + idx
        env = {
            **os.environ,
            "PORT": str(port),
            "STATIC_PORT": str(static_port),
            "RENDER_POOL_SIZE": str(self.config.pool_size),
            "RENDER_TIMEOUT_MS": str(self.config.render_timeout_ms),
            "PROTOCOL_TIMEOUT_MS": str(self.config.protocol_timeout_ms),
            "PREWARM_LIGHTING": self.config.lighting.value,
        }
        proc = await asyncio.create_subprocess_exec(
            self.config.node_binary,
            str(_RUNNER_JS),
            env=env,
            cwd=self._node_cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stdout_task = asyncio.create_task(
            self._pipe_logger(proc.stdout, f"#{idx}/stdout")
        )
        stderr_task = asyncio.create_task(
            self._pipe_logger(proc.stderr, f"#{idx}/stderr")
        )
        client = httpx.AsyncClient(timeout=self.config.request_timeout_s)
        try:
            browser_pid = await self._wait_ready_for(proc, client, port, idx)
        except BaseException:
            await self._kill_proc_group(proc, idx)
            for t in (stdout_task, stderr_task):
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass
            try:
                await client.aclose()
            except Exception:
                pass
            raise
        logger.info(f"[RENDERER] sidecar #{idx} ready on port {port}")
        return _Sidecar(idx=idx, port=port, proc=proc, client=client,
                        log_tasks=[stdout_task, stderr_task],
                        browser_pid=browser_pid)

    async def _monitor(self, sc: _Sidecar) -> None:
        """Return when the sidecar is dead: process exited or pings starve."""
        ping_url = f"http://{self.config.sidecar_host}:{sc.port}/ping"
        strikes = 0
        while True:
            try:
                await asyncio.wait_for(sc.proc.wait(), timeout=self._PING_INTERVAL_S)
                logger.warning(
                    f"[RENDERER] sidecar #{sc.idx} process exited (rc={sc.proc.returncode})"
                )
                return
            except asyncio.TimeoutError:
                pass
            try:
                r = await sc.client.get(ping_url, timeout=self._PING_TIMEOUT_S)
                ok = r.status_code == 200
            except Exception:
                ok = False
            if ok:
                strikes = 0
                sc.browser_pid = self._parse_browser_pid(r) or sc.browser_pid
                continue
            strikes += 1
            if strikes >= self._PING_STRIKES:
                logger.warning(
                    f"[RENDERER] sidecar #{sc.idx} failed {strikes} consecutive pings — restarting"
                )
                return

    async def _kill_proc_group(
        self, proc: asyncio.subprocess.Process, idx: int, browser_pid: int | None = None
    ) -> None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=self._TERM_GRACE_S)
        except asyncio.TimeoutError:
            pass
        for pgid, label in ((proc.pid, "node"), (browser_pid, "browser")):
            if pgid is None:
                continue
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                logger.warning(f"[RENDERER] sidecar #{idx} killpg({label}) denied: {exc}")
        if proc.returncode is None:
            await proc.wait()

    async def _teardown_sidecar(self, sc: _Sidecar) -> None:
        """Idempotent: every step tolerates an already-dead process, an
        already-closed client and already-finished log pumps."""
        try:
            await sc.client.aclose()
        except Exception:
            pass
        await self._kill_proc_group(sc.proc, sc.idx, sc.browser_pid)
        # Log pumps last, so the sidecar's dying words reach the logs.
        for t in sc.log_tasks:
            if not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    async def shutdown(self) -> None:
        self._shutting_down = True
        tasks = [s.task for s in self._slots if s.task is not None]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for slot in self._slots:
            sc = slot.sidecar
            slot.sidecar = None
            slot.task = None
            if sc is not None:
                await self._teardown_sidecar(sc)
        self._slots = []

    def _next_sidecar(self) -> _Sidecar | None:
        live = [s.sidecar for s in self._slots if s.sidecar is not None]
        if not live:
            return None
        sc = live[self._dispatch_counter % len(live)]
        self._dispatch_counter += 1
        return sc

    async def _post_with_retry(self, path: str, payload: dict) -> tuple[httpx.Response, int]:
        """POST to a live sidecar; on a dead-channel error retry once,
        preferring a different sidecar. Renders are idempotent, so with a
        single live sidecar retrying on the same one still beats failing.

        The try covers only the post itself — HTTP-status errors raised by
        callers after this returns are never retried.
        """
        sc = self._next_sidecar()
        if sc is None:
            raise RuntimeError("no sidecars available")
        try:
            resp = await sc.client.post(
                f"http://{self.config.sidecar_host}:{sc.port}{path}", json=payload
            )
            return resp, sc.idx
        except (*_RETRYABLE_ERRORS, RuntimeError) as exc:
            first_idx = sc.idx
            retry_sc = self._next_sidecar()
            if retry_sc is not None and retry_sc.idx == first_idx:
                alt = self._next_sidecar()
                if alt is not None and alt.idx != first_idx:
                    retry_sc = alt
            if retry_sc is None:
                raise RuntimeError("no sidecars available") from exc
            logger.warning(
                f"[RENDERER] {path} on sidecar #{first_idx} failed "
                f"({type(exc).__name__}: {exc}) — retrying on sidecar #{retry_sc.idx}"
            )
            resp = await retry_sc.client.post(
                f"http://{self.config.sidecar_host}:{retry_sc.port}{path}", json=payload
            )
            return resp, retry_sc.idx

    async def process(self, task: PipelineTask) -> PipelineTask:
        """Render the 2x2 grid (4 orbit views composed in Python)."""
        await self._render_and_fill(task, self._grid_specs())
        return task

    async def process_with_judge_views(
        self, task: PipelineTask
    ) -> tuple[dict[str, bytes], dict[str, bytes]]:
        """Grid + judge views in one sidecar request; fills the same task fields
        as ``process()`` and returns ``(white_views, gray_views)``.
        All-or-nothing: on failure both come back empty."""
        specs = self._grid_specs()
        specs += [{"name": n, "bg": self.config.judge_white_bg} for n in self._JUDGE_WHITE_VIEWS]
        specs.append({
            "name": self._JUDGE_GRAY_VIEW,
            "bg": self.config.judge_gray_bg,
            "key": self._JUDGE_GRAY_KEY,
        })
        rendered = await self._render_and_fill(task, specs)

        gray_png = rendered.pop(self._JUDGE_GRAY_KEY, None)
        gray = {self._JUDGE_GRAY_VIEW: gray_png} if gray_png else {}
        return rendered, gray

    async def render_views(
        self, task: PipelineTask, views: list[str], *, img_size: int
    ) -> dict[str, bytes]:
        """Render named preset views only, at ``img_size``, on the judge's white background.

        No grid, no mutation of the task: a probe the caller reads and discards.
        Returns an empty dict on any failure — callers treat that as "no decision".
        """
        if task.failed or (not task.js_code and not task.scene_json):
            return {}
        payload: dict = {
            "views": [{"name": v, "bg": self.config.judge_white_bg} for v in views],
            "options": {"imgSize": img_size, "lighting": self.config.lighting.value},
        }
        if task.scene_json is not None:
            payload["object"] = task.scene_json
        else:
            payload["source"] = task.js_code

        try:
            resp, sidecar_idx = await self._post_with_retry("/render/views", payload)
        except Exception as exc:  # noqa: BLE001 - a probe never fails its caller
            logger.warning(f"[RENDERER] '{task.stem}' probe FAIL (http) | {type(exc).__name__}: {exc}")
            return {}
        if resp.status_code != 200:
            logger.warning(
                f"[RENDERER] '{task.stem}' probe FAIL (status) sidecar=#{sidecar_idx} | "
                f"HTTP {resp.status_code}: {resp.text[:200] if resp.text else ''}"
            )
            return {}

        rendered: dict[str, bytes] = {}
        for name, b64 in (resp.json().get("views") or {}).items():
            try:
                rendered[name] = base64.b64decode(b64)
            except Exception as exc:  # noqa: BLE001 - skip a single bad view, keep the rest
                logger.warning(f"[RENDERER] probe view {name!r} decode failed: {exc}")
        return rendered

    def _grid_specs(self) -> list[dict]:
        bg = self.config.bg_color
        return [
            {"theta": theta, "phi": phi, "key": key, **({"bg": bg} if bg else {})}
            for (theta, phi), key in zip(GRID_VIEWS, GRID_KEYS)
        ]

    # Per-angle views consumed by the multi-stage judge. The gray front view
    # rides in the same request under a distinct key (name collides with white).
    _JUDGE_WHITE_VIEWS = (
        "front_left", "front_right", "front_below", "front_above",
        "right", "back", "left", "top_down",
    )
    _JUDGE_GRAY_VIEW = "front_left"
    _JUDGE_GRAY_KEY = "front_left_gray"

    async def _render_and_fill(self, task: PipelineTask, view_specs: list[dict]) -> dict[str, bytes]:
        """One /render/views request: grid tiles composed into ``task.rendered_png``
        (missing/invalid tile is fatal), remaining views returned."""
        if task.failed or (not task.js_code and not task.scene_json):
            logger.debug(
                f"[RENDERER] '{task.stem}' skip "
                f"(failed={task.failed}, has_code={bool(task.js_code)}, "
                f"has_scene={bool(task.scene_json)})"
            )
            return {}

        use_object = task.scene_json is not None
        payload: dict = {
            "views": view_specs,
            "options": {"imgSize": self.config.img_size, "lighting": self.config.lighting.value},
        }
        if use_object:
            payload["object"] = task.scene_json
            size_note = "scene_json"
        else:
            payload["source"] = task.js_code
            size_note = f"js_code={len(task.js_code)} bytes"

        logger.info(
            f"[RENDERER] '{task.stem}' start | views={len(view_specs)} "
            f"| path={'object' if use_object else 'code'} | {size_note}"
        )

        t0 = time.monotonic()
        try:
            resp, sidecar_idx = await self._post_with_retry("/render/views", payload)
        except Exception as exc:
            task.render_errors = [f"{type(exc).__name__}: {exc}"]
            logger.warning(
                f"[RENDERER] '{task.stem}' FAIL (http) | {task.render_errors[0]}"
            )
            return {}

        task.render_ms = (time.monotonic() - t0) * 1000.0

        if resp.status_code != 200:
            detail = resp.text[:200] if resp.text else ""
            task.render_errors = [f"HTTP {resp.status_code}: {detail}"]
            logger.warning(
                f"[RENDERER] '{task.stem}' FAIL (status) sidecar=#{sidecar_idx} | "
                f"{task.render_errors[0]} | render={task.render_ms/1000:.1f}s"
            )
            return {}

        rendered: dict[str, bytes] = {}
        for name, b64 in (resp.json().get("views") or {}).items():
            try:
                rendered[name] = base64.b64decode(b64)
            except Exception as exc:  # noqa: BLE001 - skip a single bad view, keep the rest
                logger.warning(f"[RENDERER] view {name!r} decode failed: {exc}")

        try:
            tiles = [rendered.pop(key) for key in GRID_KEYS]
            task.rendered_png = await asyncio.to_thread(
                compose_grid, tiles, img_size=self.config.img_size, gap=self.config.grid_gap
            )
        except (KeyError, ValueError) as exc:
            task.render_errors = [f"grid tile invalid: {exc}"]
            logger.warning(
                f"[RENDERER] '{task.stem}' FAIL (grid) sidecar=#{sidecar_idx} | "
                f"{task.render_errors[0]}"
            )
            return rendered

        task.refinement_rendered_pngs.append(task.rendered_png)
        logger.info(
            f"[RENDERER] '{task.stem}' PASS sidecar=#{sidecar_idx} | "
            f"png={len(task.rendered_png)}B views={len(rendered)} "
            f"render={task.render_ms/1000:.1f}s"
        )
        return rendered

    def _resolve_node_cwd(self) -> str:
        preferred = os.environ.get("NODE_CWD")
        candidates = [preferred, "/workspace", str(_RUNNER_JS.parent.parent.parent.parent)]
        for c in candidates:
            if c and os.path.isdir(os.path.join(c, "node_modules")):
                return c
        return str(_RUNNER_JS.parent)

    @staticmethod
    def _parse_browser_pid(resp: httpx.Response) -> int | None:
        try:
            pid = resp.json().get("browserPid")
            return int(pid) if pid else None
        except Exception:
            return None

    async def _wait_ready_for(
        self,
        proc: asyncio.subprocess.Process,
        client: httpx.AsyncClient,
        port: int,
        idx: int,
    ) -> int | None:
        """Wait until the sidecar answers /ping with 200; returns the reported
        chromium pid (None when the sidecar predates the reporting)."""
        deadline = time.monotonic() + self.config.startup_timeout_s
        ping_url = f"http://{self.config.sidecar_host}:{port}/ping"
        last_err: str = ""
        while time.monotonic() < deadline:
            if proc.returncode is not None:
                raise RuntimeError(
                    f"[RENDERER] sidecar #{idx} exited early: rc={proc.returncode}"
                )
            try:
                r = await client.get(ping_url, timeout=2.0)
                if r.status_code == 200:
                    return self._parse_browser_pid(r)
                last_err = f"status={r.status_code}"
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(1.0)
        raise RuntimeError(
            f"[RENDERER] sidecar #{idx} not ready after {self.config.startup_timeout_s}s "
            f"(last: {last_err})"
        )

    @staticmethod
    async def _pipe_logger(stream: asyncio.StreamReader | None, which: str) -> None:
        if stream is None:
            return
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    if "stderr" in which:
                        logger.warning(f"[renderer-sidecar {which}] {text}")
                    else:
                        logger.debug(f"[renderer-sidecar {which}] {text}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(f"[renderer-sidecar {which}] pipe error: {exc}")
