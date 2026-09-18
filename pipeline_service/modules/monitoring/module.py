import asyncio
import multiprocessing
import queue

import aiohttp
import openai
from logger_config import logger
from openai import AsyncOpenAI

from .settings import ProbeResult


def _checker_worker(
    settings,
    result_queue: multiprocessing.Queue,
    poll_interval: float = 10.0,
) -> None:
    """Runs in subprocess, probes the model service, pushes latest ProbeResult."""
    import logging
    logging.basicConfig(level=logging.INFO, force=True)
    
    async def _probe_health(health_url: str) -> bool:
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as s:
                async with s.get(health_url) as resp:
                    return resp.status == 200
        except aiohttp.ClientError:
            return False

    async def _probe(client: AsyncOpenAI, health_url: str) -> ProbeResult:
        model = settings.vllm_model_name
        if not await _probe_health(health_url):
            return ProbeResult.down(model, "health check failed")
        try:
            served = {entry.id for entry in (await client.models.list()).data}
            # Reachable is not the same as serving this model. An endpoint that serves a
            # different one answers every generation with NotFoundError while the pod
            # still reports healthy, so health has to mean that this model in particular
            # is being served.
            if model not in served:
                return ProbeResult.down(model, f"endpoint serves {sorted(served)}, not {model}")
            return ProbeResult.running(model)
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            return ProbeResult.starting(model, str(e))
        except openai.InternalServerError as e:
            return ProbeResult.down(model, str(e))
        except openai.APIStatusError as e:
            return ProbeResult.down(model, f"HTTP {e.status_code}: {e.message}")
        except Exception as e:
            return ProbeResult.down(model, f"unexpected: {e}")

    async def poll() -> None:
        health_url = settings.vllm_url.rstrip("/").rsplit("/v1", 1)[0] + "/health"
        client = AsyncOpenAI(base_url=settings.vllm_url, api_key=settings.vllm_api_key)

        while True:
            result = await _probe(client, health_url)
            # drain stale, push latest — we only care about current state
            try:
                while True:
                    result_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                result_queue.put_nowait(result)
            except queue.Full:
                pass
            logger.debug(f"Checker Status -  {result}")
            await asyncio.sleep(poll_interval)

    asyncio.run(poll())