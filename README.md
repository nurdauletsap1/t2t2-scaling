# T2T² — Train-to-Test-Time Scaling Laws

Empirical derivation of unified scaling laws of the form:

```
logit(accuracy) = A/N^α + B/D^β + C/k^γ
```

Where **N** = non-embedding parameters, **D** = training tokens, **k** = test-time samples (majority vote).

**Core hypothesis:** overtraining small models (10M–80M params) + majority voting at inference
beats Chinchilla-optimal large models by 20%+ at equal total FLOP budgets.

---

## Overview

| Component | Description |
|---|---|
| `src/model.py` | Llama-style LM builder (RoPE, RMSNorm, SwiGLU, tied embeddings) |
| `src/data.py` | GSM8K + MATH data pipeline with leakage prevention |
| `src/train.py` | Accelerate-based training loop (bf16, OOM retry, WandB) |
| `src/eval.py` | pass@k, maj@k evaluation with FLOP tracking |
| `launch_experiment.py` | Single-run entry point |
| `launch_sweep.py` | Multi-GPU parallel sweep manager |
| `generate_configs.py` | Generate 36-run sweep grid |
| `validate_setup.py` | Pre-flight environment checks |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Validate your environment

```bash
python validate_setup.py --no-wandb
```

All 16 checks must pass before running experiments.

### 2. Run a single tiny experiment (smoke test)

```bash
python launch_experiment.py --config configs/test_tiny.yaml --no-wandb
```

Results are written to `results/test_tiny_100M_s0.json`.

### 3. Run the small model config

```bash
python launch_experiment.py --config configs/test_small.yaml --no-wandb
```

---

## Full Sweep

### 1. Generate all 36 configs

```bash
python generate_configs.py
```

This creates `configs/sweep/sweep_{size}_{D}M_s{seed}.yaml` for:
- Sizes: `tiny`, `small`, `medium`, `large`
- Token budgets: `100M`, `300M`, `1000M`
- Seeds: `0`, `1`, `2`

### 2. Launch the sweep

```bash
# Uses all available GPUs automatically
python launch_sweep.py

# Skip WandB
python launch_sweep.py --no-wandb

# Custom directories
python launch_sweep.py --sweep-dir configs/sweep --results-dir results
```

The sweep manager:
- Detects GPU count via `torch.cuda.device_count()`
- Skips runs where `results/{run_id}.json` already exists
- Runs one experiment per GPU in parallel
- Prints a live status table every 10 seconds
- Exits with code 1 if any run fails

---

## Results Schema

Every completed run writes `results/{run_id}.json`:

```json
{
  "run_id": "sweep_tiny_100M_s0",
  "model_size": "tiny_3.1M",
  "model_type": "llama",
  "total_params": 16011776,
  "non_embed_params": 3145984,
  "D_tokens": 100000000,
  "k_values": [1, 4, 8, 16],
  "pass@1": 0.045,
  "pass@4": 0.071,
  "pass@8": 0.089,
  "pass@16": 0.103,
  "maj@1": 0.045,
  "maj@4": 0.083,
  "maj@8": 0.092,
  "maj@16": 0.098,
  "best_val_loss": 2.341,
  "total_train_steps": 1526,
  "consumed_tokens": 100000000,
  "compute_flops": 1.92e+15,
  "train_flops": 1.90e+15,
  "infer_flops": 2.00e+13,
  "timestamp": 1713292800.0,
  "git_commit": "a1b2c3d",
  "config_hash": "-1234567890",
  "seed": 0
}
```

---

## Reproducing Paper Results

1. Install dependencies: `pip install -r requirements.txt`
2. Set WandB entity in `configs/sweep/*.yaml` (field `wandb.entity`)
3. Generate configs: `python generate_configs.py`
4. Validate: `python validate_setup.py`
5. Run sweep: `python launch_sweep.py`

Recommended hardware: 4× A100 80GB or equivalent.
Full sweep estimated runtime: ~48–96 hours on 4 GPUs.

---

## Model Sizes

| Size | Layers | Hidden | Heads | Intermediate | ~Non-Embed Params |
|---|---|---|---|---|---|
| tiny | 4 | 256 | 4 | 1024 | ~3.1M |
| small | 6 | 384 | 6 | 1536 | ~6.3M |
| medium | 8 | 512 | 8 | 2048 | ~12.5M |
| large | 12 | 768 | 12 | 3072 | ~28M |

---

## Citation

```bibtex
@misc{t2t2-2024,
  title  = {Train-to-Test-Time Scaling Laws},
  year   = {2024},
}
```
