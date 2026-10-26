#!/usr/bin/env python3
"""runner_guards.py -- shared library of runner self-defence guards

Shared guard functions for artifact-existence checks (false completion markers), VRAM sharing between stages (OOM),
self-match false positives of pgrep, and physical VRAM checks.

Plan: see the experiment description.
Plan: see the experiment description.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

_META_CHARS = set(".*+?()[]{}|^$\\")


class GuardError(RuntimeError):
    """Guard check failure. The message contains the check name, the measured value and the required value"""


def assert_artifacts(dir: str, glob: str, min_count: int, *, recursive: bool = False) -> int:
    """Artifact-existence check. Returns the number of artifacts when it passes.
    By default only files directly under --dir are counted (non-recursive).
    A caller that needs to count artifacts in subdirectories must state recursive=True (CLI: --recursive) explicitly."""
    d = Path(dir)
    if not d.is_dir():
        raise GuardError(f"dir not found: {dir}")
    matches = d.rglob(glob) if recursive else d.glob(glob)
    n = len(list(matches))
    if n < min_count:
        msg = (f"artifacts {n} < required {min_count} "
               f"(dir={dir}, glob={glob}, recursive={recursive})")
        if not recursive:
            n_rec = len(list(d.rglob(glob)))
            if n_rec > n:
                msg += (f"; subdirectories contain {n_rec - n} more match(es) "
                        f"(recursive count={n_rec}). If the artifacts are meant to live in "
                        f"subdirectories, pass --recursive (recursive=True).")
        raise GuardError(msg)
    return n


def write_done_marker(
    dir: str, glob: str, min_count: int, marker: str = "ALL_DONE", *, recursive: bool = False
) -> Path:
    """Writes the completion marker only if the check passes (the check and the marker writing are inseparable)."""
    n = assert_artifacts(dir, glob, min_count, recursive=recursive)
    marker_path = Path(dir) / marker
    ts = datetime.datetime.now().astimezone().isoformat()
    marker_path.write_text(
        f"{ts} n={n} glob={glob} min_count={min_count} recursive={int(recursive)}\n"
    )
    return marker_path


def _parse_cuda_visible_devices(raw: str) -> Optional[List[int]]:
    """Expects comma-separated integers only. Returns None if it cannot be parsed (fallback to all GPUs)."""
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    if not parts:
        return None
    indices: List[int] = []
    for p in parts:
        if not re.fullmatch(r"-?\d+", p):
            return None
        indices.append(int(p))
    return indices


def vram_precheck(min_free_gib: float, wait_minutes: float = 0) -> float:
    """Physical VRAM free-space check by measurement with nvidia-smi."""

    def _measure() -> float:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise GuardError(f"nvidia-smi not found: {exc}") from exc

        if proc.returncode != 0:
            raise GuardError(
                f"nvidia-smi exited with code {proc.returncode}: {proc.stderr.strip()}"
            )

        stdout = proc.stdout.strip()
        if not stdout:
            raise GuardError("nvidia-smi produced no output")

        import os

        target_indices = None
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd:
            target_indices = _parse_cuda_visible_devices(cvd)
            if target_indices is None:
                print(
                    "[GUARD] CUDA_VISIBLE_DEVICES unparsable, falling back to all GPUs",
                    file=sys.stderr,
                )

        free_values_mib: List[float] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            fields = [f.strip() for f in line.split(",")]
            if len(fields) != 2:
                raise GuardError(f"nvidia-smi output unparsable: {line!r}")
            idx_str, free_str = fields
            try:
                idx = int(idx_str)
                free_mib = float(free_str)
            except ValueError as exc:
                raise GuardError(
                    f"nvidia-smi output unparsable: {line!r}"
                ) from exc

            if target_indices is not None and idx not in target_indices:
                continue
            free_values_mib.append(free_mib)

        if not free_values_mib:
            raise GuardError(
                "nvidia-smi returned no matching GPU rows "
                f"(CUDA_VISIBLE_DEVICES={cvd!r})"
            )

        return max(free_values_mib) / 1024.0

    free_gib = _measure()
    if free_gib >= min_free_gib:
        return free_gib

    if wait_minutes <= 0:
        raise GuardError(
            f"VRAM free {free_gib:.1f} GiB < required {min_free_gib}"
        )

    deadline = time.monotonic() + wait_minutes * 60.0
    while True:
        print(
            f"[GUARD] waiting VRAM: free={free_gib:.1f} GiB (need {min_free_gib})",
            file=sys.stderr,
        )
        if time.monotonic() >= deadline:
            raise GuardError(
                f"VRAM free {free_gib:.1f} GiB < required {min_free_gib} "
                f"after waiting {wait_minutes} minutes"
            )
        time.sleep(60)
        free_gib = _measure()
        if free_gib >= min_free_gib:
            return free_gib
        if time.monotonic() >= deadline:
            raise GuardError(
                f"VRAM free {free_gib:.1f} GiB < required {min_free_gib} "
                f"after waiting {wait_minutes} minutes"
            )


def alive_pattern(name: str) -> str:
    """Generates a non-self-matching pattern for pgrep -f."""
    if not name:
        raise GuardError("alive_pattern: name must not be empty")
    first = name[0]
    if first in _META_CHARS:
        raise GuardError(
            f"alive_pattern: leading char {first!r} is a regex metachar (name={name!r})"
        )
    return f"[{first}]{name[1:]}"


def record_gpu_context(phase: str, out_dir: str, job_id: Optional[str] = None) -> dict:
    """GPU co-location record. Records the GPU occupancy at the time the job runs.

    phase: "start" (call it before your own job takes VRAM) or "end".
    out_dir: destination directory of the record (the artifact directory of the calling runner).
    job_id: if not given explicitly, the environment variable SLURM_JOB_ID, otherwise "unknown".

    This function does not stop the computation: whether nvidia-smi is missing, fails or
    gives invalid output, it raises no exception, leaves what could be obtained in a dict,
    accumulates the description of the failure as strings in dict["errors"], and prints a warning to stderr
    (its behaviour intentionally differs from the GuardError raised by vram_precheck).
    """
    timestamp = time.time()
    job_id_resolved = job_id or os.environ.get("SLURM_JOB_ID") or "unknown"
    node = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    slurm_job_gpus = os.environ.get("SLURM_JOB_GPUS")

    errors: List[str] = []

    target_indices: Optional[List[int]] = None
    if cuda_visible_devices:
        target_indices = _parse_cuda_visible_devices(cuda_visible_devices)
        if target_indices is None:
            print(
                "[GUARD] record_gpu_context: CUDA_VISIBLE_DEVICES unparsable, "
                "falling back to all GPUs",
                file=sys.stderr,
            )

    # 1. per-GPU utilization/memory (device-scoped; index/name identify a GPU,
    #    not a process -- no other-user PII).
    gpus: List[dict] = []
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0:
            errors.append(
                f"nvidia-smi (query-gpu) exited with code {proc.returncode}: "
                f"{proc.stderr.strip()}"
            )
        else:
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                fields = [f.strip() for f in line.split(",")]
                if len(fields) != 5:
                    errors.append(f"nvidia-smi (query-gpu) output unparsable: {line!r}")
                    continue
                try:
                    idx = int(fields[0])
                    name = fields[1]
                    util = int(round(float(fields[2])))
                    mem_used = int(round(float(fields[3])))
                    mem_total = int(round(float(fields[4])))
                except ValueError:
                    errors.append(f"nvidia-smi (query-gpu) output unparsable: {line!r}")
                    continue
                if target_indices is not None and idx not in target_indices:
                    continue
                gpus.append(
                    {
                        "index": idx,
                        "name": name,
                        "utilization_gpu": util,
                        "memory_used_mib": mem_used,
                        "memory_total_mib": mem_total,
                    }
                )
    except FileNotFoundError as exc:
        errors.append(f"nvidia-smi (query-gpu) not found: {exc}")
    except subprocess.TimeoutExpired as exc:
        errors.append(f"nvidia-smi (query-gpu) timed out: {exc}")
    except Exception as exc:  # noqa: BLE001 - record-only, never raise
        errors.append(f"nvidia-smi (query-gpu) failed: {exc.__class__.__name__}: {exc}")

    # 2. compute-apps (PID + memory only -- never trace the command line etc. from the PIDs of other users.
    #    What is obtained here is only the PID and the amount of memory used).
    #    No exception is raised for insufficient permission, no support or empty output; it is treated as
    #    compute_apps_available=False (under SLURM cgroups empty output and insufficient permission cannot be
    #    distinguished programmatically, so empty output is not treated as confirmed zero).
    compute_apps_available = False
    compute_apps: List[dict] = []
    try:
        proc2 = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc2.returncode != 0:
            errors.append(
                f"nvidia-smi (query-compute-apps) exited with code {proc2.returncode}: "
                f"{proc2.stderr.strip()}"
            )
        else:
            lines = [ln.strip() for ln in proc2.stdout.splitlines() if ln.strip()]
            if not lines:
                errors.append(
                    "nvidia-smi (query-compute-apps) returned no rows "
                    "(indistinguishable from permission-restricted visibility)"
                )
            else:
                for line in lines:
                    fields = [f.strip() for f in line.split(",")]
                    if len(fields) != 2:
                        errors.append(
                            f"nvidia-smi (query-compute-apps) output unparsable: {line!r}"
                        )
                        continue
                    try:
                        pid = int(fields[0])
                        used_mib = int(round(float(fields[1])))
                    except ValueError:
                        errors.append(
                            f"nvidia-smi (query-compute-apps) output unparsable: {line!r}"
                        )
                        continue
                    compute_apps.append({"pid": pid, "used_gpu_memory_mib": used_mib})
                compute_apps_available = True
    except FileNotFoundError as exc:
        errors.append(f"nvidia-smi (query-compute-apps) not found: {exc}")
    except subprocess.TimeoutExpired as exc:
        errors.append(f"nvidia-smi (query-compute-apps) timed out: {exc}")
    except Exception as exc:  # noqa: BLE001 - record-only, never raise
        errors.append(f"nvidia-smi (query-compute-apps) failed: {exc.__class__.__name__}: {exc}")

    peer_memory_mib = sum(g["memory_used_mib"] for g in gpus)
    peer_process_count = len(compute_apps) if compute_apps_available else -1

    result = {
        "phase": phase,
        "timestamp": timestamp,
        "job_id": job_id_resolved,
        "node": node,
        "cuda_visible_devices": cuda_visible_devices,
        "slurm_job_gpus": slurm_job_gpus,
        "gpus": gpus,
        "compute_apps_available": compute_apps_available,
        "compute_apps": compute_apps,
        "peer_memory_mib": peer_memory_mib,
        "peer_process_count": peer_process_count,
        "errors": errors,
    }

    # 3. persistence (overwrite, not append. To avoid concurrent appends to the same path, one file
    #    per job and phase). A write failure does not stop the computation either.
    try:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        json_path = out_path / f"gpu_context_{job_id_resolved}_{phase}.json"
        json_path.write_text(json.dumps(result, indent=2))
    except Exception as exc:  # noqa: BLE001 - record-only, never raise
        result["errors"].append(f"failed to write JSON: {exc.__class__.__name__}: {exc}")

    if result["errors"]:
        print(
            f"[GUARD] record_gpu_context(phase={phase}): {len(result['errors'])} "
            f"item(s) failed: {result['errors']}",
            file=sys.stderr,
        )

    return result


# no dashboard directory exists in the public release (DASHBOARD_PROGRESS_DIR can still be set explicitly)
_PROGRESS_DIR_CANDIDATES = []

_progress_dir_warned = False


def _resolve_progress_dir() -> Optional[str]:
    """If DASHBOARD_PROGRESS_DIR is set, that; otherwise the first existing directory among
    _PROGRESS_DIR_CANDIDATES. Tests can control the candidates by replacing this module attribute
    _PROGRESS_DIR_CANDIDATES."""
    env_dir = os.environ.get("DASHBOARD_PROGRESS_DIR")
    if env_dir:
        return env_dir
    for cand in _PROGRESS_DIR_CANDIDATES:
        if os.path.isdir(cand):
            return cand
    return None


def write_progress(
    done: int,
    total: int,
    label: str = "",
    unit: str = "item",
    job_id: Optional[str] = None,
    min_interval_s: float = 30.0,
) -> Optional[Path]:
    """Writes the progress file for the cross-project dashboard.

    If no output destination is found, no exception is raised; a WARNING is printed to stderr only once per process
    and None is returned. An inconsistency of done/total
    (a bug of the caller) stops with ValueError.
    """
    global _progress_dir_warned

    if done < 0 or total <= 0 or done > total:
        raise ValueError(
            f"write_progress: invalid done/total (done={done}, total={total})"
        )

    if job_id is not None:
        resolved_job_id = job_id
    else:
        arr_job = os.environ.get("SLURM_ARRAY_JOB_ID")
        arr_task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if arr_job and arr_task:
            resolved_job_id = f"{arr_job}_{arr_task}"
        elif os.environ.get("SLURM_JOB_ID"):
            resolved_job_id = os.environ["SLURM_JOB_ID"]
        else:
            resolved_job_id = f"pid{os.getpid()}@{socket.gethostname()}"

    out_dir = _resolve_progress_dir()
    if out_dir is None:
        if not _progress_dir_warned:
            print(
                "WARNING write_progress: progress dir not found; dashboard ETA disabled",
                file=sys.stderr,
            )
            _progress_dir_warned = True
        return None

    safe_job_id = resolved_job_id.replace("/", "_")
    file_path = Path(out_dir) / f"{safe_job_id}.json"

    now = datetime.datetime.now().astimezone()
    now_iso = now.isoformat()

    existing = None
    if file_path.exists():
        try:
            existing = json.loads(file_path.read_text())
        except (OSError, ValueError):
            existing = None

    if existing is not None and done < total:
        prev_updated = None
        try:
            prev_updated = datetime.datetime.fromisoformat(existing["updated_at"])
        except (KeyError, ValueError, TypeError):
            prev_updated = None
        if prev_updated is not None:
            elapsed = (now - prev_updated).total_seconds()
            if elapsed < min_interval_s:
                return file_path

    started_at = now_iso
    if existing is not None and existing.get("started_at"):
        started_at = existing["started_at"]

    payload = {
        "schema": 1,
        "job_id": resolved_job_id,
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "label": label,
        "unit": unit,
        "done": done,
        "total": total,
        "started_at": started_at,
        "updated_at": now_iso,
    }

    tmp_path = file_path.with_name(file_path.name + f".tmp{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2))
    os.replace(tmp_path, file_path)
    return file_path


def _cmd_assert_artifacts(args: argparse.Namespace) -> int:
    n = assert_artifacts(args.dir, args.glob, args.min_count, recursive=args.recursive)
    print(f"[PASS] artifacts n={n}")
    return 0


def _cmd_write_done_marker(args: argparse.Namespace) -> int:
    marker_path = write_done_marker(
        args.dir, args.glob, args.min_count, args.marker, recursive=args.recursive
    )
    print(str(marker_path.resolve()))
    return 0


def _cmd_vram_precheck(args: argparse.Namespace) -> int:
    free_gib = vram_precheck(args.min_free_gib, args.wait_minutes)
    print(f"[PASS] vram free={free_gib:.1f} GiB")
    return 0


def _cmd_alive_pattern(args: argparse.Namespace) -> int:
    pattern = alive_pattern(args.name)
    print(pattern)
    return 0


def _cmd_write_progress(args: argparse.Namespace) -> int:
    path = write_progress(args.done, args.total, label=args.label, unit=args.unit)
    if path is None:
        print("[WARN] progress dir not found; nothing written", file=sys.stderr)
        return 0
    print(str(path.resolve()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runner_guards.py",
        description="Shared library of runner self-defence guards",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_aa = sub.add_parser("assert-artifacts", help="artifact-existence check")
    p_aa.add_argument("--dir", required=True)
    p_aa.add_argument("--glob", required=True)
    p_aa.add_argument("--min-count", required=True, type=int)
    p_aa.add_argument("--recursive", action="store_true", help="also count subdirectories recursively (default: only directly under --dir)")
    p_aa.set_defaults(func=_cmd_assert_artifacts)

    p_wd = sub.add_parser("write-done-marker", help="write the completion marker only if the check passes")
    p_wd.add_argument("--dir", required=True)
    p_wd.add_argument("--glob", required=True)
    p_wd.add_argument("--min-count", required=True, type=int)
    p_wd.add_argument("--marker", default="ALL_DONE")
    p_wd.add_argument("--recursive", action="store_true", help="also count subdirectories recursively (default: only directly under --dir)")
    p_wd.set_defaults(func=_cmd_write_done_marker)

    p_vp = sub.add_parser("vram-precheck", help="physical VRAM free-space check by measurement with nvidia-smi")
    p_vp.add_argument("--min-free-gib", required=True, type=float)
    p_vp.add_argument("--wait-minutes", default=0, type=float)
    p_vp.set_defaults(func=_cmd_vram_precheck)

    p_ap = sub.add_parser("alive-pattern", help="generate a non-self-matching pattern for pgrep")
    p_ap.add_argument("--name", required=True)
    p_ap.set_defaults(func=_cmd_alive_pattern)

    p_wp = sub.add_parser("write-progress", help="write the progress file for the dashboard")
    p_wp.add_argument("--done", required=True, type=int)
    p_wp.add_argument("--total", required=True, type=int)
    p_wp.add_argument("--label", default="")
    p_wp.add_argument("--unit", default="item")
    p_wp.set_defaults(func=_cmd_write_progress)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except GuardError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
