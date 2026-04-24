"""Entry point for a single T2T² training + evaluation run."""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml


def _set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_config(config: Dict) -> None:
    """Check all required top-level config fields exist.

    Args:
        config: Loaded YAML config dict.

    Raises:
        ValueError: If any required field is missing.
    """
    required_top = ["run_id", "model", "training", "evaluation", "seed", "output_dir", "checkpoint_dir"]
    required_model = ["size", "n_layers", "hidden_dim", "n_heads", "intermediate_dim", "max_seq_len", "vocab_size"]
    required_training = ["D_tokens", "per_gpu_batch_size"]
    required_eval = ["k_values"]

    for key in required_top:
        if key not in config:
            raise ValueError(f"Missing required config field: '{key}'")
    for key in required_model:
        if key not in config["model"]:
            raise ValueError(f"Missing required config.model field: '{key}'")
    for key in required_training:
        if key not in config["training"]:
            raise ValueError(f"Missing required config.training field: '{key}'")
    for key in required_eval:
        if key not in config["evaluation"]:
            raise ValueError(f"Missing required config.evaluation field: '{key}'")


def _config_hash(config: Dict) -> str:
    """Compute a short hash of the config dict for deduplication.

    Args:
        config: Config dict.

    Returns:
        Integer hash string.
    """
    raw = json.dumps(config, sort_keys=True, default=str)
    return str(hash(raw))


def _get_git_commit() -> str:
    """Get the current git commit hash (short).

    Returns:
        7-character commit hash, or 'unknown' if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _estimate_runtime(config: Dict, n_params: int) -> str:
    """Rough runtime estimate based on token budget and model size.

    Args:
        config: Full config dict.
        n_params: Non-embedding parameter count.

    Returns:
        Human-readable estimated runtime string.
    """
    d_tokens = config["training"]["D_tokens"]
    flops_estimate = 6 * n_params * d_tokens
    # Assume ~1e12 FLOPs/sec for a mid-range GPU
    seconds = flops_estimate / 1e12
    if seconds < 60:
        return f"~{seconds:.0f}s"
    elif seconds < 3600:
        return f"~{seconds/60:.0f}m"
    else:
        return f"~{seconds/3600:.1f}h"


def main() -> None:
    """Run a full T2T² experiment: train then evaluate."""
    parser = argparse.ArgumentParser(description="Launch a T2T² experiment.")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config file.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging.")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index to use (sets CUDA_VISIBLE_DEVICES).")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # ── Load and validate config ───────────────────────────────────────────
    config_path = args.config
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.no_wandb:
        config.setdefault("wandb", {})
        config["wandb"]["project"] = None

    _validate_config(config)

    # ── Seeds ─────────────────────────────────────────────────────────────
    _set_seeds(config["seed"])

    # ── Imports ───────────────────────────────────────────────────────────
    from src.model import build_llama_model, count_parameters
    from src.data import load_datasets
    from src.train import train
    from src.eval import evaluate

    # ── Build model ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Run ID : {config['run_id']}")
    print(f"  Config : {config_path}")
    print(f"{'='*60}\n")

    model = build_llama_model(config["model"])
    params = count_parameters(model)
    est_runtime = _estimate_runtime(config, params["non_embedding"])
    train_flops = 6 * params["non_embedding"] * config["training"]["D_tokens"]

    print(f"Model summary:")
    print(f"  Total params      : {params['total']:,}")
    print(f"  Non-embedding     : {params['non_embedding']:,}")
    print(f"  Embedding params  : {params['embedding']:,}")
    print(f"  Est. train FLOPs  : {train_flops:.3e}")
    print(f"  Est. runtime      : {est_runtime}")
    print()

    # ── Load datasets ─────────────────────────────────────────────────────
    print("Loading datasets (leakage check included)...")
    train_dataset, val_dataset, test_dataset = load_datasets(config)
    print(f"  Train: {len(train_dataset):,} examples")
    print(f"  Val  : {len(val_dataset):,} examples")
    print(f"  Test : {len(test_dataset):,} examples\n")

    # ── Train ─────────────────────────────────────────────────────────────
    print("Starting training...")
    train_results = train(config, model, train_dataset, val_dataset)

    print(f"\nTraining complete:")
    print(f"  Best val loss    : {train_results['best_val_loss']:.4f}")
    print(f"  Total steps      : {train_results['total_steps']:,}")
    print(f"  Consumed tokens  : {train_results['consumed_tokens']:,}")
    print(f"  Checkpoint       : {train_results['checkpoint_path']}\n")

    # ── Load best checkpoint ──────────────────────────────────────────────
    best_ckpt_path = Path(train_results["checkpoint_path"])
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        # Rebuild model to ensure no compile wrapper issues
        model = build_llama_model(config["model"])
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best checkpoint from {best_ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("Starting evaluation...")
    k_values = config["evaluation"].get("k_values", [1, 4, 8, 16])
    eval_results = evaluate(config, model, test_dataset, k_values=k_values)

    print("\nEvaluation results:")
    for k in k_values:
        print(f"  pass@{k} = {eval_results[f'pass@{k}']:.4f}   maj@{k} = {eval_results[f'maj@{k}']:.4f}")
    print()

    # ── Assemble result JSON ───────────────────────────────────────────────
    size_label = f"{config['model']['size']}_{params['non_embedding']//1_000_000:.1f}M"
    compute_flops = eval_results["compute_flops"]

    result = {
        "run_id": config["run_id"],
        "model_size": size_label,
        "model_type": "llama",
        "total_params": params["total"],
        "non_embed_params": params["non_embedding"],
        "D_tokens": config["training"]["D_tokens"],
        "k_values": k_values,
        **{f"pass@{k}": eval_results[f"pass@{k}"] for k in k_values},
        **{f"maj@{k}": eval_results[f"maj@{k}"] for k in k_values},
        "best_val_loss": train_results["best_val_loss"],
        "total_train_steps": train_results["total_steps"],
        "consumed_tokens": train_results["consumed_tokens"],
        "compute_flops": compute_flops["total_flops"],
        "train_flops": compute_flops["train_flops"],
        "infer_flops": compute_flops["infer_flops"],
        "timestamp": time.time(),
        "git_commit": _get_git_commit(),
        "config_hash": _config_hash(config),
        "seed": config["seed"],
    }

    # ── Save results ──────────────────────────────────────────────────────
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{config['run_id']}.json"

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {result_path}")
    print(f"\n{'='*60}")
    print(f"  Run {config['run_id']} complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
