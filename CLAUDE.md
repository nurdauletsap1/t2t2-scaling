# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Environment validation (run before anything else)
python3 validate_setup.py --no-wandb

# Single experiment
python3 launch_experiment.py --config configs/test_tiny.yaml --no-wandb
python3 launch_experiment.py --config configs/test_small.yaml --no-wandb

# Generate the 36-run sweep grid
python3 generate_configs.py

# Run the full sweep across all available GPUs
python3 launch_sweep.py --no-wandb

# Quick import checks (use these to test each src/ module in isolation)
python3 -c "from src.model import build_llama_model, count_parameters; import yaml; cfg = yaml.safe_load(open('configs/test_tiny.yaml')); m = build_llama_model(cfg['model']); print(count_parameters(m))"
python3 -c "from src.data import load_datasets"
python3 -c "from src.train import train"
python3 -c "from src.eval import evaluate"
```

No test suite exists yet. The canonical correctness check for `src/model.py` is:
```bash
python3 -c "
from src.model import build_llama_model, count_parameters
import yaml
cfg = yaml.safe_load(open('configs/test_tiny.yaml'))
model = build_llama_model(cfg['model'])
params = count_parameters(model)
assert 3_000_000 < params['non_embedding'] < 4_000_000, params
print('model.py ✓', params)
"
```

## Architecture

The experiment pipeline is a linear chain: config → model → data → train → eval → result JSON.

**Entry point:** `launch_experiment.py` is the only file that writes to disk (results JSON). It owns seed-setting and result serialization. All `src/` functions are pure — no side effects on import, no disk I/O except checkpoints inside `train()`.

**Config schema:** Every run is driven by a YAML file. The full config dict is passed through to every `src/` function. Key top-level sections: `model`, `training`, `evaluation`, `wandb`, plus `run_id`, `seed`, `output_dir`, `checkpoint_dir`.

**`src/model.py`:** `build_llama_model(config["model"])` → `LlamaForCausalLM`. The `intermediate_dim` field in YAML follows the "4× hidden" convention; internally the code applies `actual_intermediate = int(2/3 * intermediate_dim)` to produce the SwiGLU hidden size (this is standard Llama practice — a 3-matrix SwiGLU with dimension 2/3 × X has the same param count as a 2-matrix MLP with dimension X). Changing this formula changes param counts.

**`src/data.py`:** `load_datasets(config)` downloads GSM8K + MATH, interleaves them 50/50 by index, does a SHA-256 leakage check (raises `ValueError` on overlap), takes a 10% seeded validation split, and returns three `MathDataset` instances. The test set is strictly MATH test split only. Tokenizer is GPT-2 (vocab=50257), padding token = EOS token.

**`src/train.py`:** `train(config, model, train_dataset, val_dataset)`. Uses `Accelerator` for bf16 + DDP. Gradient accumulation is computed to target 65,536 tokens/step. Checkpoints go to `checkpoints/{run_id}/step_{N:08d}.pt` and contain model + optimizer + scheduler state. Auto-resumes from the latest checkpoint in that directory on startup. WandB logging falls back to `logs/wandb_fallback.jsonl` on failure.

**`src/eval.py`:** `evaluate(config, model, test_dataset, k_values)`. Generates `max(k_values)` samples per problem, then computes pass@k and maj@k by slicing the first k. Answer extraction priority: `\boxed{}` → `$$...$$` → `$...$` → last number. Numeric answers compared with tolerance 1e-3, symbolic with exact normalized string match. Returns a dict including a `compute_flops` sub-dict.

**`generate_configs.py`** / **`launch_sweep.py`:** The sweep is 4 sizes × 3 D_token budgets × 3 seeds = 36 runs. `launch_sweep.py` assigns one config per GPU, skips runs where `results/{run_id}.json` already exists, and polls every 10 seconds.

## Key Invariants

- `intermediate_dim` in YAML is NOT passed directly to `LlamaConfig.intermediate_size`; it is multiplied by 2/3 first. If you add new model size configs, keep `intermediate_dim = 4 × hidden_dim`.
- `count_parameters` subtracts `vocab_size × hidden_dim` (tied) or `2 × vocab_size × hidden_dim` (untied) from total. Non-embedding params are used in all FLOP calculations.
- Training FLOPs: `6 × N_nonembedding × D_tokens`. Inference FLOPs: `2 × N_nonembedding × k × mean_tokens_generated × n_problems`.
- `launch_experiment.py` is the only writer of `results/*.json`. Do not write results from `src/` files.
- Seeds are set once in `launch_experiment.py` via `_set_seeds(config["seed"])` and never inside `src/`.
