"""Batch processing wrapper around the single-file `privacy-blur` pipeline.

Scans a work directory laid out as

    {work_dir}/
        {folder_a}/extracted/{folder_a}-sensor-{1..4}.mkv
        {folder_b}/extracted/{folder_b}-sensor-{1..4}.mkv
        ...
        results.json

and for every existing sensor file:

1.  Runs the detect + blur pipeline once (re-using the YOLO/TRT engines so
    we don't pay the ~10 s warmup per file).
2.  Writes the blurred output to a sibling temp file
    `{name}.blur.tmp{ext}` so a crashed encode never corrupts the original.
3.  On success, atomically replaces the original via `os.replace` (or moves
    the original to `{name}{ext}.orig` first when `--keep-original` is set).
4.  Records progress + per-file stats in `{work_dir}/results.json`. The
    JSON is written after each file so an interrupted batch resumes cleanly.

Status machine for a job: `pending` -> `running` -> `done` | `failed`.
`--force` re-runs `done` jobs, `--retry-failed` re-runs `failed` jobs.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
import traceback
from pathlib import Path

from .cli import add_common_args, build_detectors, process_video
from .utils import choose_device


RESULTS_FILENAME = "results.json"
RESULTS_VERSION = 1
SENSOR_INDICES = (1, 2, 3, 4)


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _job_id(folder: str, sensor: int) -> str:
    return f"{folder}/sensor-{sensor}"


def scan_folders(work_dir: Path) -> list[dict]:
    """Discover `{folder}/extracted/{folder}-sensor-{1..4}.mkv` files.

    Returns one dict per existing file, in deterministic (folder, sensor)
    order. Folders without an `extracted/` subdirectory are silently skipped.
    """
    jobs: list[dict] = []
    for sub in sorted(p for p in work_dir.iterdir() if p.is_dir()):
        extracted = sub / "extracted"
        if not extracted.is_dir():
            continue
        for i in SENSOR_INDICES:
            f = extracted / f"{sub.name}-sensor-{i}.mkv"
            if f.is_file():
                jobs.append({
                    "id": _job_id(sub.name, i),
                    "folder": sub.name,
                    "sensor": i,
                    "input": f.relative_to(work_dir).as_posix(),
                })
    return jobs


def load_results(work_dir: Path) -> dict:
    """Read `results.json` if present, else return an empty document."""
    path = work_dir / RESULTS_FILENAME
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and isinstance(data.get("jobs"), list):
                return data
            print(f"[WARN] {path} has unexpected shape; starting fresh.", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Could not parse {path}: {e}. Starting fresh.", file=sys.stderr)
    return {
        "version": RESULTS_VERSION,
        "work_dir": str(work_dir),
        "updated_at": _now_iso(),
        "jobs": [],
    }


def save_results(work_dir: Path, results: dict) -> None:
    """Atomically write `results.json` (write tmp + rename) so a crash mid-write
    can't leave a half-written file."""
    results["updated_at"] = _now_iso()
    path = work_dir / RESULTS_FILENAME
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    os.replace(str(tmp), str(path))


def merge_jobs(results: dict, discovered: list[dict]) -> tuple[int, int]:
    """Add newly-discovered jobs into `results['jobs']` without disturbing the
    state of jobs that already exist. Returns (new_count, total_count)."""
    existing = {j["id"]: j for j in results["jobs"]}
    new = 0
    for d in discovered:
        if d["id"] in existing:
            existing[d["id"]]["input"] = d["input"]
            continue
        results["jobs"].append({
            **d,
            "status": "pending",
            "frames": None,
            "elapsed_s": None,
            "fps": None,
            "n_boxes_total": None,
            "per_frame_ms": None,
            "started_at": None,
            "finished_at": None,
            "error": None,
        })
        new += 1
    return new, len(results["jobs"])


def needs_run(job: dict, force: bool, retry_failed: bool) -> bool:
    s = job.get("status", "pending")
    if force:
        return True
    if s == "done":
        return False
    if s == "failed":
        return retry_failed
    return True  # pending, running (stale), or unknown


def _temp_output(input_path: Path) -> Path:
    """`/foo/bar/x.mkv` -> `/foo/bar/x.blur.tmp.mkv`. Same directory keeps
    `os.replace` atomic (same filesystem)."""
    return input_path.parent / f"{input_path.stem}.blur.tmp{input_path.suffix}"


def _process_one(args, job: dict, plate, face, work_dir: Path,
                 results: dict, keep_original: bool) -> None:
    """Process a single job. Mutates `job` (status/stats/timestamps) and writes
    `results.json` at the running/done transitions so progress is durable."""
    input_abs = (work_dir / job["input"]).resolve()
    temp_abs = _temp_output(input_abs)

    if not input_abs.is_file():
        raise FileNotFoundError(f"Input file missing: {input_abs}")

    # Remove any leftover temp from a previous crashed run; otherwise ffmpeg
    # will refuse or, worse, silently append.
    if temp_abs.exists():
        try:
            temp_abs.unlink()
        except Exception as e:
            raise RuntimeError(f"Could not remove stale temp file {temp_abs}: {e}")

    args.input = str(input_abs)
    args.output = str(temp_abs)

    job["status"] = "running"
    job["started_at"] = _now_iso()
    job["finished_at"] = None
    job["error"] = None
    save_results(work_dir, results)

    stats = process_video(args, plate, face)

    if keep_original:
        backup = input_abs.with_suffix(input_abs.suffix + ".orig")
        if backup.exists():
            input_abs.unlink()
        else:
            os.replace(str(input_abs), str(backup))
    os.replace(str(temp_abs), str(input_abs))

    job["status"] = "done"
    job["frames"] = stats["frames"]
    job["elapsed_s"] = round(stats["elapsed_s"], 3)
    job["fps"] = round(stats["fps"], 3)
    job["n_boxes_total"] = stats["n_boxes_total"]
    job["per_frame_ms"] = {k: round(v, 2) for k, v in stats["per_frame_ms"].items()}
    job["finished_at"] = _now_iso()
    job["error"] = None


def main():
    ap = argparse.ArgumentParser(
        "privacy-blur-batch",
        description=(
            "Batch-process a work directory of sensor recordings. "
            "Looks for {work-dir}/{folder}/extracted/{folder}-sensor-{1..4}.mkv "
            "and replaces each with its blurred version in place. "
            "Progress is tracked in {work-dir}/results.json so a stopped run "
            "resumes where it left off."
        ),
    )
    ap.add_argument("--work-dir", required=True,
                    help="Root directory containing per-trip folders.")
    ap.add_argument("--scan-only", action="store_true",
                    help="Discover files and update results.json without processing.")
    ap.add_argument("--force", action="store_true",
                    help="Re-process even jobs that are already marked 'done'.")
    ap.add_argument("--retry-failed", action="store_true",
                    help="Re-process previously failed jobs.")
    ap.add_argument("--keep-original", action="store_true",
                    help="Keep the source as {name}.mkv.orig before replacing in place.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would run; do not process or modify files.")
    add_common_args(ap)
    args = ap.parse_args()

    work_dir = Path(args.work_dir).resolve()
    if not work_dir.is_dir():
        sys.exit(f"--work-dir does not exist or is not a directory: {work_dir}")

    discovered = scan_folders(work_dir)
    results = load_results(work_dir)
    results["work_dir"] = str(work_dir)
    new_count, total = merge_jobs(results, discovered)
    save_results(work_dir, results)

    done = sum(1 for j in results["jobs"] if j.get("status") == "done")
    failed = sum(1 for j in results["jobs"] if j.get("status") == "failed")
    pending = [j for j in results["jobs"] if needs_run(j, args.force, args.retry_failed)]

    print(f"[INFO] Work dir: {work_dir}")
    print(f"[INFO] Discovered on disk: {len(discovered)} sensor files ({new_count} new)")
    print(f"[INFO] Jobs total/done/failed/to-run: {total}/{done}/{failed}/{len(pending)}")

    if args.scan_only:
        print("[OK] Scan complete (--scan-only).")
        return

    if args.dry_run:
        for j in pending:
            print(f"[DRY] would process {j['input']}")
        return

    if not pending:
        print("[OK] Nothing to do.")
        return

    # process_video reads args.input/args.output; they're overwritten per job.
    args.input = ""
    args.output = ""

    device = choose_device(args.device)
    print(f"[INFO] Using device: {device}")
    plate, face = build_detectors(args, device)

    batch_start = time.perf_counter()
    for idx, job in enumerate(pending, start=1):
        print(f"\n[BATCH {idx}/{len(pending)}] {job['id']}  ({job['input']})")
        try:
            _process_one(args, job, plate, face, work_dir, results,
                         keep_original=args.keep_original)
            save_results(work_dir, results)
        except KeyboardInterrupt:
            # Leave the job pending so the next run picks it up, and clean
            # up any half-written temp file.
            job["status"] = "pending"
            job["error"] = "interrupted"
            _cleanup_temp(work_dir, job)
            save_results(work_dir, results)
            print("\n[INFO] Interrupted. Progress saved.")
            sys.exit(130)
        except Exception as e:
            job["status"] = "failed"
            job["error"] = f"{type(e).__name__}: {e}"
            job["finished_at"] = _now_iso()
            traceback.print_exc()
            _cleanup_temp(work_dir, job)
            save_results(work_dir, results)
            print(f"[FAIL] {job['id']}: {job['error']}")

    elapsed = time.perf_counter() - batch_start
    final_done = sum(1 for j in results["jobs"] if j.get("status") == "done")
    final_failed = sum(1 for j in results["jobs"] if j.get("status") == "failed")
    print(
        f"\n[BATCH] Finished in {elapsed:.1f}s | "
        f"done={final_done} failed={final_failed} total={len(results['jobs'])}"
    )


def _cleanup_temp(work_dir: Path, job: dict) -> None:
    try:
        temp_abs = _temp_output((work_dir / job["input"]).resolve())
        if temp_abs.exists():
            temp_abs.unlink()
    except Exception:
        pass


if __name__ == "__main__":
    main()
