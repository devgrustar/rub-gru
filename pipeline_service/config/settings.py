from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings

from modules.js_checker.settings import JSCheckerConfig
from modules.judge.embedder_settings import EmbedderConfig
from modules.renderer.settings import RendererConfig

_here = Path(__file__).parent.parent
config_file_dir = _here / "configuration.yaml"
if not config_file_dir.exists():
    config_file_dir = _here.parent / "configuration.yaml"



class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 10006
    debug: bool = False


class CoderProbeConfig(BaseModel):
    """Warm-up coder throughput probe (runs once when the models are up, before the warmup task).

    Sends `concurrency` text-only code requests to the coder and measures aggregate output tok/s.
    The batch wall-clock is coder tokens / tok/s, so this is the one number that predicts whether the
    audit host fits the 7200 s regeneration budget. With min_tps > 0 and replacements remaining, a slower
    pod requests REPLACE; min_tps = 0 only logs. Calibrate on a healthy 4xH200 pod first and set min_tps
    to ~0.8 x the probe value measured there.
    """
    enabled: bool = True
    concurrency: int = 48
    max_tokens: int = 512
    min_tps: float = 0.0
    timeout_s: float = 300.0


class PreflightConfig(BaseModel):
    """Optional host-gating thresholds for modules/metrics/preflight.py (per GPU family defaults apply when None).
    Read by preflight.py directly from the YAML; declared here so the service accepts the key."""
    min_tflops: float | None = None
    min_stream48_gbps: float | None = None
    min_power_limit_w: float | None = None
    min_download_mbps: float | None = None


class PipelineConfig(BaseModel):
    batch_time_budget: float = 1800.0
    prompt_timeout: float = 120.0
    use_planner: bool = True
    render_from_object: bool = False
    # When false, disables the whole refinement loop: the critic and every
    # patcher/repair iteration are skipped. Only iteration-0 generation runs
    # (multigen + judge bracket when ensemble_size > 1, else a single coder pass).
    refinement_enabled: bool = True
    coder_probe: CoderProbeConfig = Field(default_factory=CoderProbeConfig)
    # Added to the round seed for every candidate (seed = round_seed + seed_offset + k). Give each of our
    # hotkeys a different offset so two miners never sample the same seed stream: byte-identical outputs
    # shared across hotkeys are a source-audit ban criterion.
    seed_offset: int = 0


class VllmServeConfig(BaseModel):
    """
    Options for launching `vllm serve` via run.sh (local endpoints only).
    """

    model: str = ""
    revision: str | None = None
    port: int | None = None
    gpu_ids: str = "auto"
    tensor_parallel_size: int = 0
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    max_num_seqs: int = 4
    api_key: str = "local"
    # Per-client vLLM launch overrides (consumed by llm/spawn.py). Let a client
    # run from its own venv with a different transformers — e.g. GLM-4.6V needs a
    # newer transformers than the Qwen coder.
    vllm_bin: str | None = None              # default /opt/vllm-env/bin/vllm
    trust_remote_code: bool = False
    reasoning_parser: str | None = "qwen3"   # null/"" -> omit --reasoning-parser
    extra_args: list[str] = Field(default_factory=list)


class ProviderRoutingConfig(BaseModel):
    """OpenRouter `provider` routing block — mirrors the upstream API.

    Forward-compat: `extra="allow"` lets newer OpenRouter fields (e.g.
    `max_price`, `preferred_max_latency`) pass through without bumping
    this model.
    """

    model_config = ConfigDict(extra="allow")

    order: list[str] | None = None
    only: list[str] | None = None
    ignore: list[str] | None = None
    allow_fallbacks: bool | None = True
    sort: str | None = None


class LLMClientConfig(BaseModel):
    """One named client endpoint."""
    base_url: str
    api_key_env: str = ""
    api_key: str = ""
    enabled: bool = False
    vllm: VllmServeConfig | None = None

    @property
    def backend(self) -> str:
        """Detect backend type from base_url."""
        _LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0")
        url = self.base_url.lower()
        return "vllm" if any(h in url for h in _LOCAL_HOSTS) else "openrouter"

class ActorConfig(BaseModel):
    """Per-actor config."""
    explain: bool = False  # judge only: run the opponent-independent 'explain' GLM call (fills detail['issues']; no effect on the verdict)

    workers: int = 1
    queue_size: int = 8
    client: str | None = None
    extra_clients: list[str] = Field(default_factory=list)
    model: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    seed: int = 42
    reasoning_effort: str | None = None
    enable_thinking: bool | None = None
    ensemble_size: int = 1
    ensemble_temperature: float = 0.3
    multimodal: bool = False
    providers: ProviderRoutingConfig | None = None

class ActorsConfig(BaseModel):
    planner: ActorConfig = ActorConfig(
        workers=2, queue_size=8,
        client="openrouter", model="qwen/qwen2.5-vl-72b-instruct",
    )
    coder: ActorConfig = ActorConfig(
        workers=2, queue_size=8,
        client="openrouter", model="qwen/qwen-2.5-72b-instruct",
        max_tokens=8192,
    )
    patcher: ActorConfig = ActorConfig(workers=2, queue_size=8)
    critic: ActorConfig = ActorConfig(
        workers=3, queue_size=8,
        client="openrouter", model="qwen/qwen2.5-vl-72b-instruct",
    )
    judge: ActorConfig = ActorConfig(
        workers=4, queue_size=8,
        client="openrouter", model="qwen/qwen2.5-vl-72b-instruct",
        max_tokens=1024,
    )
    checker: ActorConfig = ActorConfig(workers=2, queue_size=8)
    renderer: ActorConfig = ActorConfig(workers=1, queue_size=4)


_LLM_ACTORS_BASE: tuple[str, ...] = ("coder", "critic")
_LLM_ACTORS_WITH_PLANNER: tuple[str, ...] = ("planner",) + _LLM_ACTORS_BASE


class EventBusConfig(BaseModel):
    max_iter: int = 2
    task_deadline_s: float = 60.0
    score_threshold: float = 0.80

def _default_llm_clients() -> dict[str, LLMClientConfig]:
    """Default LLM clients."""
    _DEFAULT_VISION = "http://localhost:8001/v1"
    _DEFAULT_CODER = "http://localhost:8002/v1"
    _DEFAULT_OPENROUTER = "https://openrouter.ai/api/v1"

    return {
        "openrouter": LLMClientConfig(
            base_url=_DEFAULT_OPENROUTER, api_key_env="OPENROUTER_API_KEY", enabled=True,
        ),
        "vision": LLMClientConfig(
            base_url=_DEFAULT_VISION, api_key_env="VLLM_API_KEY", enabled=True,
        ),
        "coder": LLMClientConfig(
            base_url=_DEFAULT_CODER, api_key_env="VLLM_API_KEY", enabled=True,
        ),
    }


class SettingsConf(BaseSettings):
    api: APIConfig = APIConfig()
    pipeline: PipelineConfig = PipelineConfig()
    preflight: PreflightConfig = PreflightConfig()
    benchmark: bool = True
    warmup: bool = True
    llm_clients: dict[str, LLMClientConfig] = Field(default_factory=_default_llm_clients)
    actors: ActorsConfig = ActorsConfig()
    event_bus: EventBusConfig = EventBusConfig()
    js_checker: JSCheckerConfig = JSCheckerConfig()
    renderer: RendererConfig = RendererConfig()
    embedder: EmbedderConfig = EmbedderConfig()

    @model_validator(mode="after")
    def _validate_pure_image_requires_multimodal(self) -> "SettingsConf":
        """When the planner is disabled the coder must consume the reference
        image directly, so multimodal=true is required on a vision-capable model."""
        if not self.pipeline.use_planner and not self.actors.coder.multimodal:
            raise ValueError(
                "pipeline.use_planner=false requires actors.coder.multimodal=true "
                "(coder must consume the reference image directly when there is no "
                "planner). Set actors.coder.multimodal=true and choose a "
                "vision-capable model."
            )
        return self

    @model_validator(mode="after")
    def _validate_actor_clients(self) -> "SettingsConf":
        known = set(self.llm_clients.keys())
        actor_names = list(
            _LLM_ACTORS_WITH_PLANNER if self.pipeline.use_planner
            else _LLM_ACTORS_BASE
        )
        if self.actors.coder.ensemble_size > 1:
            actor_names.append("judge")
        for name in actor_names:
            actor: ActorConfig = getattr(self.actors, name)
            if actor.client is None:
                raise ValueError(
                    f"actors.{name}.client is required (must name an entry in llm_clients). "
                    f"Known clients: {sorted(known)}"
                )
            if actor.client not in known:
                raise ValueError(
                    f"actors.{name}.client={actor.client!r} does not exist in llm_clients. "
                    f"Known clients: {sorted(known)}"
                )
            if not actor.model:
                raise ValueError(
                    f"actors.{name}.model is required (non-empty string)."
                )
            client_cfg = self.llm_clients[actor.client]
            if not client_cfg.enabled:
                raise ValueError(
                    f"actors.{name}.client={actor.client!r} points to a disabled client "
                    f"(llm_clients.{actor.client}.enabled=false). Enable the client "
                    f"or change actors.{name}.client."
                )
            for extra in actor.extra_clients:
                if extra not in known:
                    raise ValueError(
                        f"actors.{name}.extra_clients entry {extra!r} does not exist "
                        f"in llm_clients. Known clients: {sorted(known)}"
                    )
                if not self.llm_clients[extra].enabled:
                    raise ValueError(
                        f"actors.{name}.extra_clients entry {extra!r} points to a "
                        f"disabled client (llm_clients.{extra}.enabled=false)."
                    )
        return self

    @model_validator(mode="after")
    def _validate_llm_client_topology(self) -> "SettingsConf":
        """Client topology: either 'openrouter' in llm_clients, or local vision+coder."""
        clients = self.llm_clients
        if "openrouter" in clients:
            return self
        if "vision" not in clients or "coder" not in clients:
            raise ValueError(
                "You must have both vision and coder clients enabled."
            )
        for label in ("vision", "coder"):
            cfg = clients[label]
            if not cfg.enabled:
                raise ValueError(
                    f"llm_clients.{label}.enabled=false is not allowed."
                )
            vllm = cfg.vllm
            if vllm is None or not (vllm.model or "").strip():
                raise ValueError(
                    f"llm_clients.{label}: set vllm.model (HF repo id) so that run.sh can start the vLLM server."
                )
        return self


def _load_yml_config(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Config not found: {path}") from error


data_yaml = _load_yml_config(config_file_dir)
settings = SettingsConf.model_validate(data_yaml)
