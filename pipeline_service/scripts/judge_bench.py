"""Judge throughput bench: S1-shaped requests (reference + two views) against a running judge vLLM.

Run from the directory that holds configuration.yaml (the container's /workspace), e.g.
    python pipeline_service/scripts/judge_bench.py --n 128 --concurrency 32 --img-size 1024
Compares img sizes or vLLM flags in minutes instead of a full 32-prompt batch. Reports req/s, prompt and
completion tok/s, latency percentiles and mean prompt_tokens (about 4.2k per S1 request at 1024 px, 1.1k at 518).
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import random
import sys
import time
from enum import Enum
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from PIL import Image, ImageDraw
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.judge.multi_stage import (  # noqa: E402
    S1_ANGLE_DESC,
    S4_PER_ANGLE_PROMPT,
    S4_SYSTEM_PROMPT,
    S4_VLM_MAX_TOKENS,
    PenaltyResponse,
    SideGuardVerdict,
    _parse_or_repair,
    _s1_messages,
)


class Shape(str, Enum):
    S1 = "s1"
    S4 = "s4"


class RequestResult(BaseModel):
    ok: bool
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    parsed: bool = False
    error: str | None = None


class BenchReport(BaseModel):
    shape: Shape
    img_size: int
    ref_size: int
    n: int
    concurrency: int
    distinct_views: int
    wall_s: float
    req_per_s: float
    prompt_tok_per_s: float
    completion_tok_per_s: float
    mean_prompt_tokens: float
    latency_p50_s: float
    latency_p90_s: float
    latency_max_s: float
    parse_failures: int
    errors: int
    first_error: str | None = None


def _synthetic_png(size: int, seed: int) -> bytes:
    """Random filled shapes on white: cheap stand-in for a rendered view of the same pixel budget."""
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(6, 14)):
        color = tuple(rng.randint(0, 220) for _ in range(3))
        kind = rng.choice(("ellipse", "rectangle", "polygon"))
        if kind == "polygon":
            draw.polygon([(rng.randint(0, size), rng.randint(0, size)) for _ in range(rng.randint(3, 6))], fill=color)
        else:
            x0, y0 = rng.randint(0, size - 2), rng.randint(0, size - 2)
            box = (x0, y0, rng.randint(x0 + 1, size), rng.randint(y0 + 1, size))
            getattr(draw, kind)(box, fill=color)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _data_url(png: bytes) -> str:
    import base64
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _s4_messages(front_url: str, angle_url: str, angle_desc: str) -> list[dict]:
    return [
        {"role": "system", "content": S4_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "Front view of the 3D model:"},
            {"type": "image_url", "image_url": {"url": front_url}},
            {"type": "text", "text": f"Same 3D model viewed from {angle_desc}:"},
            {"type": "image_url", "image_url": {"url": angle_url}},
            {"type": "text", "text": S4_PER_ANGLE_PROMPT.format(angle_desc=angle_desc)},
        ]},
    ]


def _build_request(shape: Shape, i: int, ref_url: str, views: list[str]) -> tuple[list[dict], type[BaseModel], int]:
    left, right = views[(2 * i) % len(views)], views[(2 * i + 1) % len(views)]
    if shape is Shape.S1:
        angle_desc = list(S1_ANGLE_DESC.values())[i % len(S1_ANGLE_DESC)]
        return _s1_messages(ref_url, left, right, angle_desc), PenaltyResponse, 1024
    return _s4_messages(left, right, ("the right side", "the back", "the left side", "directly above")[i % 4]), SideGuardVerdict, S4_VLM_MAX_TOKENS


async def _one(client: AsyncOpenAI, model: str, messages: list[dict], schema: type[BaseModel], max_tokens: int,
               seed: int, extra_body: dict | None) -> RequestResult:
    t0 = time.monotonic()
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            seed=seed,
            response_format={"type": "json_schema", "json_schema": {"name": "bench", "schema": schema.model_json_schema()}},
            extra_body=extra_body or {},
        )
    except Exception as exc:  # noqa: BLE001 - every failure mode is a data point here
        return RequestResult(ok=False, latency_s=time.monotonic() - t0, error=f"{type(exc).__name__}: {str(exc)[:160]}")
    usage = completion.usage
    text = completion.choices[0].message.content or ""
    return RequestResult(
        ok=True,
        latency_s=time.monotonic() - t0,
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        parsed=_parse_or_repair(text, schema) is not None,
    )


async def _burst(client, model, shape, count, concurrency, ref_url, views, seed, extra_body, offset=0) -> tuple[list[RequestResult], float]:
    sem = asyncio.Semaphore(concurrency)

    async def guarded(i: int) -> RequestResult:
        messages, schema, max_tokens = _build_request(shape, i, ref_url, views)
        async with sem:
            return await _one(client, model, messages, schema, max_tokens, seed + i, extra_body)

    t0 = time.monotonic()
    results = await asyncio.gather(*(guarded(offset + i) for i in range(count)))
    return results, time.monotonic() - t0


def _report(shape, args, results: list[RequestResult], wall: float) -> BenchReport:
    ok = [r for r in results if r.ok]
    lat = sorted(r.latency_s for r in results) or [0.0]
    prompt_toks = sum(r.prompt_tokens for r in ok)
    errors = [r.error for r in results if r.error]
    return BenchReport(
        shape=shape, img_size=args.img_size, ref_size=args.ref_size, n=len(results),
        concurrency=args.concurrency, distinct_views=args.distinct_views, wall_s=round(wall, 1),
        req_per_s=round(len(ok) / wall, 2) if wall else 0.0,
        prompt_tok_per_s=round(prompt_toks / wall) if wall else 0.0,
        completion_tok_per_s=round(sum(r.completion_tokens for r in ok) / wall) if wall else 0.0,
        mean_prompt_tokens=round(prompt_toks / len(ok)) if ok else 0.0,
        latency_p50_s=round(lat[len(lat) // 2], 2),
        latency_p90_s=round(lat[min(len(lat) - 1, int(0.9 * len(lat)))], 2),
        latency_max_s=round(lat[-1], 2),
        parse_failures=sum(1 for r in ok if not r.parsed),
        errors=len(errors),
        first_error=errors[0] if errors else None,
    )


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8002/v1")
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--model", default="zai-org/GLM-4.6V-Flash")
    ap.add_argument("--shape", type=Shape, choices=list(Shape), default=Shape.S1)
    ap.add_argument("--n", type=int, default=128, help="timed requests")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=None, help="untimed requests first (default: --concurrency)")
    ap.add_argument("--img-size", type=int, default=1024, help="edge of every synthetic view")
    ap.add_argument("--ref-size", type=int, default=1024, help="edge of the synthetic reference (ignored with --ref)")
    ap.add_argument("--ref", type=Path, default=None, help="real reference image instead of a synthetic one")
    ap.add_argument("--distinct-views", type=int, default=None,
                    help="distinct view images to cycle through (1 = maximal cache hits; default 2*n = none)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chat-template-kwargs", type=json.loads, default=None,
                    help='JSON forwarded as extra_body.chat_template_kwargs, e.g. \'{"enable_thinking": false}\'')
    ap.add_argument("--json", type=Path, default=None, help="write the report as JSON here")
    args = ap.parse_args()
    args.distinct_views = args.distinct_views or 2 * args.n
    warmup = args.concurrency if args.warmup is None else args.warmup

    ref_png = args.ref.read_bytes() if args.ref else _synthetic_png(args.ref_size, args.seed)
    ref_url = _data_url(ref_png)
    views = [_data_url(_synthetic_png(args.img_size, args.seed + 1 + i)) for i in range(args.distinct_views)]
    extra_body = {"chat_template_kwargs": args.chat_template_kwargs} if args.chat_template_kwargs else None

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key,
                         timeout=httpx.Timeout(900.0, connect=30.0), max_retries=0)
    if warmup:
        w, w_wall = await _burst(client, args.model, args.shape, warmup, args.concurrency, ref_url, views, args.seed, extra_body)
        print(f"warm-up: {sum(1 for r in w if r.ok)}/{len(w)} ok in {w_wall:.1f}s (not scored)")
    results, wall = await _burst(client, args.model, args.shape, args.n, args.concurrency, ref_url, views,
                                 args.seed, extra_body, offset=warmup)
    report = _report(args.shape, args, results, wall)
    for key, value in report.model_dump().items():
        print(f"{key:>20}: {value}")
    if args.json:
        args.json.write_text(report.model_dump_json(indent=1))
        print(f"report -> {args.json}")
    return 0 if report.errors == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
