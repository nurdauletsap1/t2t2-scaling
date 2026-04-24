"""Generate the 36-run sweep configuration files for T2T² experiments."""

from pathlib import Path
from typing import Dict

import yaml


MODEL_SPECS: Dict[str, Dict] = {
    "tiny": {
        "size": "tiny",
        "n_layers": 4,
        "hidden_dim": 256,
        "n_heads": 4,
        "intermediate_dim": 1024,
        "max_seq_len": 2048,
        "vocab_size": 50257,
        "tie_embeddings": True,
    },
    "small": {
        "size": "small",
        "n_layers": 6,
        "hidden_dim": 384,
        "n_heads": 6,
        "intermediate_dim": 1536,
        "max_seq_len": 2048,
        "vocab_size": 50257,
        "tie_embeddings": True,
    },
    "medium": {
        "size": "medium",
        "n_layers": 8,
        "hidden_dim": 512,
        "n_heads": 8,
        "intermediate_dim": 2048,
        "max_seq_len": 2048,
        "vocab_size": 50257,
        "tie_embeddings": True,
    },
    "large": {
        "size": "large",
        "n_layers": 12,
        "hidden_dim": 768,
        "n_heads": 12,
        "intermediate_dim": 3072,
        "max_seq_len": 2048,
        "vocab_size": 50257,
        "tie_embeddings": True,
    },
}

D_TOKENS_LIST = [100_000_000, 300_000_000, 1_000_000_000]
SEEDS = [0, 1, 2]


def _make_config(size: str, d_tokens: int, seed: int) -> Dict:
    """Build a full config dict for a single sweep run.

    Args:
        size: Model size key ('tiny', 'small', 'medium', 'large').
        d_tokens: Training token budget.
        seed: Random seed.

    Returns:
        Complete config dict ready to serialize as YAML.
    """
    d_label = f"{d_tokens // 1_000_000:.0f}M"
    run_id = f"sweep_{size}_{d_label}_s{seed}"

    lr = 5e-4 if size in ("tiny", "small") else 3e-4
    return {
        "run_id": run_id,
        "model": MODEL_SPECS[size],
        "training": {
            "D_tokens": d_tokens,
            "per_gpu_batch_size": 8,
            "learning_rate": lr,
            "weight_decay": 0.1,
            "betas": [0.9, 0.95],
            "warmup_steps": 100,
            "val_every_steps": 500,
            "early_stopping_patience": 200,
            "bf16": True,
            "compile": True,
        },
        "evaluation": {
            "k_values": [1, 4, 8, 16],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_new_tokens": 512,
        },
        "wandb": {
            "project": "t2t2-scaling",
            "entity": None,
        },
        "seed": seed,
        "output_dir": "results/",
        "checkpoint_dir": "checkpoints/",
    }


def generate_all_configs(sweep_dir: Path) -> list:
    """Generate all 36 sweep config YAML files.

    Args:
        sweep_dir: Directory to write configs into.

    Returns:
        List of (run_id, path) tuples for all generated configs.
    """
    sweep_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    for size in ["tiny", "small", "medium", "large"]:
        for d_tokens in D_TOKENS_LIST:
            for seed in SEEDS:
                cfg = _make_config(size, d_tokens, seed)
                d_label = f"{d_tokens // 1_000_000:.0f}M"
                filename = f"sweep_{size}_{d_label}_s{seed}.yaml"
                path = sweep_dir / filename
                with open(path, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                generated.append((cfg["run_id"], path))

    return generated


def main() -> None:
    """Generate configs and print a summary table."""
    sweep_dir = Path("configs/sweep")
    generated = generate_all_configs(sweep_dir)

    print(f"\nGenerated {len(generated)} sweep configs in {sweep_dir}/\n")
    col_w = [30, 10, 12, 6]
    header = f"{'run_id':<{col_w[0]}} {'D_tokens':>{col_w[1]}} {'size':>{col_w[2]}} {'seed':>{col_w[3]}}"
    print(header)
    print("-" * sum(col_w))

    for run_id, path in generated:
        parts = run_id.split("_")
        size = parts[1]
        d_str = parts[2]
        seed = parts[3]
        print(f"{run_id:<{col_w[0]}} {d_str:>{col_w[1]}} {size:>{col_w[2]}} {seed:>{col_w[3]}}")

    print(f"\nTotal: {len(generated)} configs ({len(['tiny','small','medium','large'])} sizes × "
          f"{len(D_TOKENS_LIST)} D values × {len(SEEDS)} seeds)")


if __name__ == "__main__":
    main()
