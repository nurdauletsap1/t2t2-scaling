"""Parallel multi-GPU sweep manager for T2T² experiments."""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class JobState:
    """Tracks the state of a single experiment job."""

    run_id: str
    config_path: Path
    gpu_id: int
    proc: Optional[subprocess.Popen] = None
    start_time: Optional[float] = None
    status: str = "queued"  # queued | running | done | failed | skipped


def _get_run_id_from_config(config_path: Path) -> str:
    """Extract run_id from a sweep YAML filename (without loading YAML).

    Args:
        config_path: Path to the YAML config file.

    Returns:
        run_id string derived from the filename stem.
    """
    # e.g. sweep_tiny_100M_s0.yaml → sweep_tiny_100M_s0
    return config_path.stem


def _result_exists(run_id: str, results_dir: Path) -> bool:
    """Check whether a result file already exists for this run.

    Args:
        run_id: Experiment identifier.
        results_dir: Directory containing result JSON files.

    Returns:
        True if the result file exists.
    """
    return (results_dir / f"{run_id}.json").exists()


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as a human-readable string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def _print_status_table(jobs: List[JobState], now: float) -> None:
    """Print a live status table for all active and recent jobs.

    Args:
        jobs: List of all job states.
        now: Current timestamp (time.time()).
    """
    running = [j for j in jobs if j.status == "running"]
    if not running:
        return

    header = f"{'GPU':>4} | {'run_id':<35} | {'status':<8} | {'elapsed':<10}"
    print("\n" + header)
    print("-" * len(header))
    for j in running:
        elapsed = _format_elapsed(now - j.start_time) if j.start_time else "-"
        print(f"{j.gpu_id:>4} | {j.run_id:<35} | {j.status:<8} | {elapsed:<10}")


def run_sweep(
    sweep_dir: Path,
    results_dir: Path,
    no_wandb: bool = False,
) -> None:
    """Run all configs in sweep_dir in parallel across available GPUs.

    Args:
        sweep_dir: Directory containing YAML sweep configs.
        results_dir: Directory to check for existing results.
        no_wandb: If True, pass --no-wandb to each experiment.
    """
    try:
        import torch
        n_gpus = torch.cuda.device_count()
    except ImportError:
        n_gpus = 0

    if n_gpus == 0:
        print("Warning: no GPUs detected — running on CPU (very slow).")
        n_gpus = 1
        gpu_ids = [None]
    else:
        gpu_ids = list(range(n_gpus))
        print(f"Detected {n_gpus} GPU(s): {gpu_ids}")

    config_paths = sorted(sweep_dir.glob("sweep_*.yaml"))
    if not config_paths:
        print(f"No sweep configs found in {sweep_dir}. Run generate_configs.py first.")
        sys.exit(1)

    # Build job queue, skip completed runs
    queue: List[JobState] = []
    skipped = 0
    for cfg_path in config_paths:
        run_id = _get_run_id_from_config(cfg_path)
        if _result_exists(run_id, results_dir):
            print(f"  [skip] {run_id} — result already exists")
            skipped += 1
            continue
        queue.append(JobState(run_id=run_id, config_path=cfg_path, gpu_id=-1))

    print(f"\nQueued {len(queue)} experiments, skipped {skipped}.\n")

    # Allocate GPUs
    free_gpus = list(gpu_ids)
    active_jobs: Dict[int, JobState] = {}  # gpu_id → job
    all_jobs: List[JobState] = []
    succeeded = 0
    failed = 0

    queue_iter = iter(queue)
    exhausted = False

    while True:
        # Launch new jobs on free GPUs
        while free_gpus and not exhausted:
            try:
                job = next(queue_iter)
            except StopIteration:
                exhausted = True
                break

            gpu_id = free_gpus.pop(0)
            job.gpu_id = gpu_id if gpu_id is not None else 0

            cmd = [
                sys.executable, "launch_experiment.py",
                "--config", str(job.config_path),
            ]
            if gpu_id is not None:
                cmd += ["--gpu", str(gpu_id)]
            if no_wandb:
                cmd += ["--no-wandb"]

            import os
            env = os.environ.copy()
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            job.proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            job.start_time = time.time()
            job.status = "running"
            active_jobs[job.gpu_id] = job
            all_jobs.append(job)
            print(f"  [start] GPU {gpu_id} → {job.run_id}")

        if not active_jobs:
            break

        # Poll active jobs
        time.sleep(10)
        now = time.time()

        finished_gpus = []
        for gpu_id, job in list(active_jobs.items()):
            ret = job.proc.poll()
            if ret is not None:
                if ret == 0:
                    job.status = "done"
                    succeeded += 1
                    elapsed = _format_elapsed(now - job.start_time)
                    print(f"  [done]  GPU {gpu_id} → {job.run_id} ({elapsed})")
                else:
                    job.status = "failed"
                    failed += 1
                    # Drain stdout for error context
                    out, _ = job.proc.communicate()
                    print(
                        f"  [FAIL]  GPU {gpu_id} → {job.run_id} (exit {ret})\n"
                        f"          Last output: {out[-500:] if out else '(none)'}"
                    )
                finished_gpus.append(gpu_id)

        for gpu_id in finished_gpus:
            del active_jobs[gpu_id]
            free_gpus.append(gpu_id)

        _print_status_table(list(active_jobs.values()), now)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Sweep complete.")
    print(f"  Succeeded : {succeeded}")
    print(f"  Failed    : {failed}")
    print(f"  Skipped   : {skipped}")
    print(f"{'='*50}\n")

    if failed > 0:
        sys.exit(1)


def main() -> None:
    """Parse arguments and launch the sweep."""
    parser = argparse.ArgumentParser(description="Run the full T2T² sweep across GPUs.")
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("configs/sweep"),
        help="Directory containing sweep YAML configs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory with existing result JSON files (to skip).",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging.")
    args = parser.parse_args()

    run_sweep(
        sweep_dir=args.sweep_dir,
        results_dir=args.results_dir,
        no_wandb=args.no_wandb,
    )


if __name__ == "__main__":
    main()
