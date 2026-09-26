"""Download every completed task's candidate export (code, renders, judge inputs) from a running miner.

    python pipeline_service/scripts/dump_candidates.py --base-url http://<pod-host>:10006 --out ./candidates_batch1
Each task lands in <out>/<stem>/ with manifest.json, reference image, k??.js and renders/ (see
pipeline/candidate_export.py). --no-renders fetches code and manifests only.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import httpx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:10006")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-renders", action="store_true", help="skip PNGs and embeddings (code + manifest only)")
    ap.add_argument("--keep-zip", action="store_true", help="keep the downloaded zips next to the extracted folders")
    ap.add_argument("--timeout", type=float, default=600.0, help="per-request timeout in seconds")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with httpx.Client(base_url=args.base_url, timeout=httpx.Timeout(args.timeout, connect=30.0)) as client:
        listing = client.get("/debug/tasks").raise_for_status().json()
        stems = [t["stem"] for t in listing.get("tasks", []) if t.get("status") != "in_progress"]
        skipped = len(listing.get("tasks", [])) - len(stems)
        print(f"miner status={listing.get('status')} | completed tasks={len(stems)} | in progress={skipped}")
        total_bytes = 0
        for i, stem in enumerate(stems, 1):
            zip_path = args.out / f"{stem}.zip"
            with client.stream("GET", f"/debug/tasks/{stem}/candidates.zip",
                               params={"renders": "false" if args.no_renders else "true"}) as resp:
                resp.raise_for_status()
                with zip_path.open("wb") as fh:
                    for chunk in resp.iter_bytes():
                        fh.write(chunk)
            size = zip_path.stat().st_size
            total_bytes += size
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(args.out / stem)
                files = len(zf.namelist())
            if not args.keep_zip:
                zip_path.unlink()
            print(f"[{i}/{len(stems)}] {stem}: {files} files, {size / 1e6:.1f} MB")
        print(f"done: {len(stems)} tasks, {total_bytes / 1e9:.2f} GB -> {args.out}")
    return 0 if stems else 1


if __name__ == "__main__":
    sys.exit(main())
