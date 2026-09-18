from __future__ import annotations

from typing import Any

import httpx

from config.settings import SettingsConf
from llm.balancer import LoadBalancedClient
from llm.session_store import SessionStore
from logger_config import logger
from modules.critic.agent import CriticAgent
from modules.js_checker.module import JSCheckerModule
from modules.judge.agent import JudgeAgent
from modules.judge.dino import DinoEmbedder
from modules.renderer.module import RendererModule
from modules.scene_coder.agent import SceneCoderAgent
from modules.scene_planner.agent import ScenePlannerAgent
from pipeline.orchestrator import Pipeline


def build_pipeline(
    *,
    settings: SettingsConf,
    clients: dict[str, Any],
    js_checker: JSCheckerModule,
    renderer: RendererModule,
    http_client: httpx.AsyncClient,
    session_store: SessionStore | None = None,
) -> Pipeline:
    """Wire agents + Pipeline from settings. Agents read their own actor config
    from the `settings` singleton — only the runtime `clients` table is wired in."""
    session_store = session_store or SessionStore()
    actors = settings.actors
    policy = settings.event_bus
    use_planner = settings.pipeline.use_planner
    ensemble_size = actors.coder.ensemble_size

    if use_planner:
        planner: ScenePlannerAgent | None = ScenePlannerAgent(
            clients[actors.planner.client], session_store=session_store, settings=actors.planner,
        )
    else:
        planner = None

    coder_client = clients[actors.coder.client]
    extra_names = [n for n in actors.coder.extra_clients if n in clients]
    if extra_names:
        names = [actors.coder.client, *extra_names]

        def _weight(name: str) -> int:
            vllm = settings.llm_clients[name].vllm
            return max(1, vllm.tensor_parallel_size if vllm else 1)

        weights = [_weight(n) for n in names]
        coder_client = LoadBalancedClient([clients[n] for n in names], weights)
        logger.info(
            f"Coder load-balanced across {names} | weights={weights}"
        )

    coder = SceneCoderAgent(coder_client, session_store=session_store, settings=actors.coder)
    critic = CriticAgent(clients[actors.critic.client], settings=actors.critic)

    if ensemble_size > 1:
        judge: JudgeAgent | None = JudgeAgent(
            clients[actors.judge.client], settings=actors.judge,
        )
        embedder: DinoEmbedder | None = (
            DinoEmbedder(settings.embedder) if settings.embedder.enabled else None
        )
    else:
        judge = None
        embedder = None

    return Pipeline(
        planner=planner,
        coder=coder,
        critic=critic,
        judge=judge,
        embedder=embedder,
        js_checker=js_checker,
        renderer=renderer,
        session_store=session_store,
        http_client=http_client,
        coder_multimodal=actors.coder.multimodal,
        use_planner=use_planner,
        coder_ensemble_size=ensemble_size,
        coder_ensemble_temperature=actors.coder.ensemble_temperature,
        render_from_object=settings.pipeline.render_from_object,
        seed_offset=settings.pipeline.seed_offset,
        refinement_enabled=settings.pipeline.refinement_enabled,
        max_iter=policy.max_iter,
        score_threshold=policy.score_threshold,
        task_deadline_s=policy.task_deadline_s,
        planner_limit=actors.planner.workers if use_planner else 1,
        coder_limit=actors.coder.workers,
        js_checker_limit=actors.checker.workers,
        renderer_limit=actors.renderer.workers,
        critic_limit=actors.critic.workers,
        judge_limit=actors.judge.workers,
    )
