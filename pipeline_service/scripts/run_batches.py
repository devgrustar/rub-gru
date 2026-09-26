"""Drive a miner through a prompt file in fixed-size batches and archive everything each batch produces.

    python pipeline_service/scripts/run_batches.py --base-url http://<pod>:<port> \
        --prompts tests/prompts/testset_128.txt --batch-size 32 --out generated_outputs/runpod_k50

Per batch: wait for READY/COMPLETE, POST /generate, poll /status, then save under <out>/batch_NN/:
    results.zip + results/          winner modules as the validator receives them (/results)
    tasks.json                      /debug/tasks listing (per-task metrics, timings, failures)
    tasks/<stem>.json, <stem>.png   /debug/tasks/<stem> with the base64 grid decoded to a file
    candidates/<stem>/              full candidate export (manifest, reference, every k??.js, renders, embeddings)
    pipeline.log                    /debug/logs
    summary.json                    timings and fetch errors
The next batch is submitted only after the archive is complete (a new /generate clears the pod's task state).
Prompt lines are "<stem> <url>" or a bare URL (stem = file name without extension). Resume with --start-batch.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import httpx

TERMINAL_GENERATION = {"complete"}
SUBMITTABLE = {"ready", "complete"}


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def parse_prompts(path: Path) -> list[dict]:
    prompts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            stem, url = parts
        else:
            url = parts[0]
            stem = Path(urlparse(url).path).stem
        prompts.append({"stem": stem, "image_url": url})
    return prompts


class Driver:
    def __init__(self, base_url: str, poll_s: float, request_timeout_s: float) -> None:
        self.client = httpx.Client(base_url=base_url, timeout=httpx.Timeout(request_timeout_s, connect=30.0))
        self.poll_s = poll_s

    def get(self, path: str, retries: int = 3, **kw) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(retries):
            try:
                return self.client.get(path, **kw)
            except httpx.HTTPError as exc:
                last = exc
                log(f"  GET {path} failed ({attempt + 1}/{retries}): {exc!r}")
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"GET {path} failed after {retries} attempts: {last!r}")

    def status(self) -> dict:
        return self.get("/status").json()

    def wait_for(self, wanted: set[str], timeout_s: float) -> dict:
        deadline = time.time() + timeout_s
        last = None
        while time.time() < deadline:
            try:
                st = self.status()
            except Exception as exc:
                log(f"  status unreachable: {exc!r}")
                time.sleep(self.poll_s)
                continue
            key = (st["status"], st["progress"], st["total"])
            if key != last:
                log(f"  status={st['status']} progress={st['progress']}/{st['total']}")
                last = key
            if st["status"] in wanted:
                return st
            time.sleep(self.poll_s)
        raise TimeoutError(f"pod did not reach {sorted(wanted)} within {timeout_s:.0f}s (last: {last})")

    def submit(self, prompts: list[dict], seed: int) -> int:
        r = self.client.post("/generate", json={"prompts": prompts, "seed": seed})
        r.raise_for_status()
        return int(r.json()["accepted"])

    def poll_generation(self, timeout_s: float, counts_every_s: float = 600.0) -> dict:
        t0 = time.time()
        deadline = t0 + timeout_s
        last = None
        next_counts = t0 + counts_every_s
        st: dict = {}
        while time.time() < deadline:
            try:
                st = self.status()
            except Exception as exc:
                log(f"  status unreachable: {exc!r}")
                time.sleep(self.poll_s)
                continue
            key = (st["status"], st["progress"], st["total"])
            if key != last:
                log(f"  [{time.time() - t0:6.0f}s] status={st['status']} progress={st['progress']}/{st['total']}")
                last = key
            if st["status"] in TERMINAL_GENERATION:
                return st
            if st["status"] in {"ready", "warming_up"} and time.time() - t0 > 3 * self.poll_s:
                log(f"  pod left GENERATING without COMPLETE (status={st['status']}): pod restart?")
                return st
            if time.time() >= next_counts:
                try:
                    counts = self.get("/debug/tasks").json().get("counts")
                    log(f"  [{time.time() - t0:6.0f}s] debug counts={counts}")
                except Exception as exc:
                    log(f"  /debug/tasks failed: {exc!r}")
                next_counts = time.time() + counts_every_s
            time.sleep(self.poll_s)
        log(f"  generation timeout after {timeout_s:.0f}s (status={st.get('status')})")
        return st

    def download(self, path: str, target: Path, **kw) -> int:
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.client.stream("GET", path, **kw) as resp:
            resp.raise_for_status()
            with target.open("wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
        return target.stat().st_size

    def archive(self, batch_dir: Path, stems: list[str], task_json: bool) -> dict:
        report: dict = {"errors": [], "bytes": 0, "fetch_started": time.time()}

        def attempt(label: str, fn) -> None:
            for i in range(3):
                try:
                    fn()
                    return
                except Exception as exc:
                    log(f"  {label} failed ({i + 1}/3): {exc!r}")
                    time.sleep(5 * (i + 1))
            report["errors"].append(label)

        def results() -> None:
            n = self.download("/results", batch_dir / "results.zip")
            report["bytes"] += n
            with zipfile.ZipFile(batch_dir / "results.zip") as zf:
                zf.extractall(batch_dir / "results")
                log(f"  results.zip: {len(zf.namelist())} files, {n / 1e6:.1f} MB")
        attempt("results", results)

        def listing() -> None:
            data = self.get("/debug/tasks").json()
            (batch_dir / "tasks.json").write_text(json.dumps(data, indent=1))
            report["counts"] = data.get("counts")
            log(f"  tasks.json: counts={data.get('counts')}")
        attempt("tasks.json", listing)

        for i, stem in enumerate(stems, 1):
            def candidates(stem=stem) -> None:
                zip_path = batch_dir / "candidates" / f"{stem}.zip"
                n = self.download(f"/debug/tasks/{stem}/candidates.zip", zip_path, params={"renders": "true"})
                report["bytes"] += n
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(batch_dir / "candidates" / stem)
                    files = len(zf.namelist())
                zip_path.unlink()
                log(f"  [{i}/{len(stems)}] candidates {stem[:12]}: {files} files, {n / 1e6:.1f} MB")
            attempt(f"candidates:{stem}", candidates)

            if task_json:
                def task(stem=stem) -> None:
                    r = self.get(f"/debug/tasks/{stem}")
                    r.raise_for_status()
                    data = r.json()
                    tasks_dir = batch_dir / "tasks"
                    tasks_dir.mkdir(parents=True, exist_ok=True)
                    grid = data.pop("rendered_png_b64", None)
                    if grid:
                        (tasks_dir / f"{stem}.png").write_bytes(base64.b64decode(grid))
                    data.pop("multigen_pngs_b64", None)
                    data.pop("refinement_rendered_pngs_b64", None)
                    (tasks_dir / f"{stem}.json").write_text(json.dumps(data, indent=1))
                    report["bytes"] += len(r.content)
                attempt(f"task:{stem}", task)

        def logs() -> None:
            n = self.download("/debug/logs", batch_dir / "pipeline.log")
            report["bytes"] += n
            log(f"  pipeline.log: {n / 1e6:.1f} MB")
        attempt("pipeline.log", logs)

        report["fetch_s"] = round(time.time() - report.pop("fetch_started"), 1)
        return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42, help="batch i uses seed + i")
    ap.add_argument("--start-batch", type=int, default=0, help="resume from this batch index")
    ap.add_argument("--attach-batch", type=int, default=None,
                    help="this batch index is already running on the pod: skip submit, just poll and archive it")
    ap.add_argument("--stop-after", type=int, default=None, help="stop after archiving this batch index")
    ap.add_argument("--poll", type=float, default=30.0, help="status poll interval, seconds")
    ap.add_argument("--ready-timeout", type=float, default=4 * 3600)
    ap.add_argument("--batch-timeout", type=float, default=14400 + 600)
    ap.add_argument("--request-timeout", type=float, default=1800.0)
    ap.add_argument("--no-task-json", action="store_true",
                    help="skip /debug/tasks/<stem> (large: carries every grid as base64)")
    args = ap.parse_args()

    prompts = parse_prompts(args.prompts)
    batches = [prompts[i:i + args.batch_size] for i in range(0, len(prompts), args.batch_size)]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run.json").write_text(json.dumps({
        "base_url": args.base_url, "prompts_file": str(args.prompts), "n_prompts": len(prompts),
        "batch_size": args.batch_size, "n_batches": len(batches), "seed": args.seed,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=1))
    log(f"{len(prompts)} prompts -> {len(batches)} batches of {args.batch_size}; out={args.out}")

    driver = Driver(args.base_url, args.poll, args.request_timeout)
    for idx in range(args.start_batch, len(batches)):
        batch = batches[idx]
        stems = [p["stem"] for p in batch]
        seed = args.seed + idx
        batch_dir = args.out / f"batch_{idx:02d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / "prompts.json").write_text(json.dumps(batch, indent=1))
        summary: dict = {"batch": idx, "seed": seed, "stems": stems}

        t_submit = time.time()
        if idx == args.attach_batch:
            summary["submitted"] = "attached"
            log(f"=== batch {idx + 1}/{len(batches)}: attached to the batch already running on the pod ===")
        else:
            log(f"=== batch {idx + 1}/{len(batches)}: waiting for READY ===")
            driver.wait_for(SUBMITTABLE, args.ready_timeout)
            accepted = driver.submit(batch, seed)
            summary["submitted"] = time.strftime("%Y-%m-%d %H:%M:%S")
            t_submit = time.time()
            log(f"=== batch {idx + 1}/{len(batches)}: submitted {accepted} prompts, seed={seed} ===")

        final = driver.poll_generation(args.batch_timeout)
        summary["final_status"] = final
        summary["generation_s"] = round(time.time() - t_submit, 1)
        log(f"=== batch {idx + 1}/{len(batches)}: {final.get('status')} after {summary['generation_s']:.0f}s, archiving ===")

        summary["archive"] = driver.archive(batch_dir, stems, task_json=not args.no_task_json)
        (batch_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        log(
            f"=== batch {idx + 1}/{len(batches)}: archived {summary['archive']['bytes'] / 1e9:.2f} GB "
            f"in {summary['archive']['fetch_s']:.0f}s, errors={summary['archive']['errors']} ==="
        )
        if args.stop_after is not None and idx >= args.stop_after:
            log(f"=== stopping after batch {idx + 1} as requested (--stop-after) ===")
            return 0
    log("=== all batches done ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("interrupted")
        sys.exit(130)
