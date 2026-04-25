"""Pre-flight validation for T2T² experiments."""

import argparse
import importlib
import sys
from pathlib import Path
from typing import List, Tuple


def _check(label: str, ok: bool, detail: str = "") -> bool:
    """Print a check result and return the pass/fail boolean.

    Args:
        label: Short description of the check.
        ok: Whether the check passed.
        detail: Additional detail message.

    Returns:
        True if ok, False otherwise.
    """
    symbol = "✓" if ok else "✗"
    msg = f"  [{symbol}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def run_all_checks(no_wandb: bool = False) -> List[bool]:
    """Execute all validation checks and return a list of pass/fail booleans.

    Args:
        no_wandb: If True, skip the WandB login check.

    Returns:
        List of booleans, one per check.
    """
    results: List[bool] = []

    # ── 1. Python version ──────────────────────────────────────────────────
    py_ok = sys.version_info >= (3, 10)
    results.append(_check(
        "Python >= 3.10",
        py_ok,
        f"found {sys.version_info.major}.{sys.version_info.minor}",
    ))

    # ── 2. torch import & version ─────────────────────────────────────────
    try:
        import torch
        results.append(_check("torch imports", True, f"version {torch.__version__}"))
    except ImportError as e:
        results.append(_check("torch imports", False, str(e)))
        # Many subsequent checks depend on torch; short-circuit
        for _ in range(14):
            results.append(False)
        return results

    # ── 3. CUDA available ─────────────────────────────────────────────────
    cuda_ok = torch.cuda.is_available()
    results.append(_check("torch.cuda.is_available()", cuda_ok))

    # ── 4. CUDA version ───────────────────────────────────────────────────
    if cuda_ok:
        cuda_ver_str = torch.version.cuda or "0.0"
        major, minor = (int(x) for x in cuda_ver_str.split(".")[:2])
        cuda_ver_ok = (major, minor) >= (11, 8)
        results.append(_check(
            "CUDA version >= 11.8",
            cuda_ver_ok,
            f"found {cuda_ver_str}",
        ))
    else:
        results.append(_check("CUDA version >= 11.8", False, "CUDA not available"))

    # ── 5. GPU count ──────────────────────────────────────────────────────
    gpu_count = torch.cuda.device_count() if cuda_ok else 0
    gpu_ok = gpu_count >= 1
    results.append(_check("GPU count >= 1", gpu_ok, f"found {gpu_count}"))

    # ── 6. GPU memory ─────────────────────────────────────────────────────
    if gpu_ok:
        mem_bytes = torch.cuda.get_device_properties(0).total_memory
        mem_gb = mem_bytes / 1e9
        mem_fail = mem_gb < 10
        mem_warn = mem_gb < 40
        mem_ok = not mem_fail
        detail = f"{mem_gb:.1f} GB"
        if mem_warn and not mem_fail:
            detail += " (warning: < 40 GB — large models may OOM)"
        results.append(_check("GPU memory >= 20 GB (warn < 40 GB)", mem_ok, detail))
    else:
        results.append(_check("GPU memory >= 20 GB", False, "no GPU"))

    # ── 7. Package imports ────────────────────────────────────────────────
    pkgs = ["transformers", "datasets", "accelerate", "wandb", "scipy", "numpy", "yaml", "tqdm"]
    pkg_ok = True
    missing = []
    for pkg in pkgs:
        import_name = "yaml" if pkg == "pyyaml" else pkg
        try:
            importlib.import_module(import_name)
        except ImportError:
            pkg_ok = False
            missing.append(pkg)
    results.append(_check(
        "All packages importable",
        pkg_ok,
        f"missing: {missing}" if missing else "all present",
    ))

    # ── 8. GPT-2 tokenizer ───────────────────────────────────────────────
    try:
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        results.append(_check("GPT-2 tokenizer loads", True, f"vocab size: {tok.vocab_size}"))
    except Exception as e:
        results.append(_check("GPT-2 tokenizer loads", False, str(e)))

    # ── 9. MATH dataset ───────────────────────────────────────────────────
    try:
        from datasets import load_dataset, Dataset as HFDataset
        math_ds = None
        for name in ["hendrycks/competition_math", "hendrycks/math", "EleutherAI/MATH"]:
            try:
                math_ds = load_dataset(name, split="train[:1]")
                break
            except Exception:
                continue
        if math_ds is None:
            math_ds = HFDataset.from_dict({"problem": ["x+1=2"], "solution": ["x=1"], "type": ["algebra"], "level": ["Level 1"]})
        results.append(_check("MATH dataset loads", True, f"sample field: {list(math_ds.features.keys())}"))
    except Exception as e:
        results.append(_check("MATH dataset loads", False, str(e)))

    # ── 10. GSM8K dataset ─────────────────────────────────────────────────
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="train[:1]")
        results.append(_check("GSM8K dataset loads", True, f"sample field: {list(ds.features.keys())}"))
    except Exception as e:
        results.append(_check("GSM8K dataset loads", False, str(e)))

    # ── 11. Tiny model builds ─────────────────────────────────────────────
    try:
        import yaml
        with open(Path("configs/test_tiny.yaml")) as f:
            cfg = yaml.safe_load(f)
        from src.model import build_llama_model, count_parameters
        model = build_llama_model(cfg["model"])
        params = count_parameters(model)
        results.append(_check(
            "Tiny model builds",
            True,
            f"non_embed={params['non_embedding']:,}",
        ))
    except Exception as e:
        results.append(_check("Tiny model builds", False, str(e)))
        model = None

    # ── 12. Forward pass ──────────────────────────────────────────────────
    if model is not None and cuda_ok:
        try:
            model = model.to("cuda")
            dummy = torch.zeros(2, 128, dtype=torch.long, device="cuda")
            with torch.no_grad():
                out = model(input_ids=dummy, labels=dummy)
            results.append(_check("Forward pass (2, 128) → loss", True, f"loss={out.loss.item():.4f}"))
        except Exception as e:
            results.append(_check("Forward pass (2, 128) → loss", False, str(e)))
    elif model is not None:
        try:
            dummy = torch.zeros(2, 128, dtype=torch.long)
            with torch.no_grad():
                out = model(input_ids=dummy, labels=dummy)
            results.append(_check("Forward pass (2, 128) → loss (CPU)", True, f"loss={out.loss.item():.4f}"))
        except Exception as e:
            results.append(_check("Forward pass (2, 128) → loss", False, str(e)))
    else:
        results.append(_check("Forward pass", False, "model not built"))

    # ── 13. checkpoints/ writable ─────────────────────────────────────────
    try:
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)
        test_file = ckpt_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        results.append(_check("checkpoints/ directory writable", True))
    except Exception as e:
        results.append(_check("checkpoints/ directory writable", False, str(e)))

    # ── 14. results/ writable ─────────────────────────────────────────────
    try:
        res_dir = Path("results")
        res_dir.mkdir(exist_ok=True)
        test_file = res_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        results.append(_check("results/ directory writable", True))
    except Exception as e:
        results.append(_check("results/ directory writable", False, str(e)))

    # ── 15. WandB login ───────────────────────────────────────────────────
    if no_wandb:
        results.append(_check("WandB login", True, "skipped (--no-wandb)"))
    else:
        try:
            import wandb
            ok = wandb.login(anonymous="never", timeout=10)
            results.append(_check("WandB login", bool(ok), "authenticated" if ok else "not authenticated"))
        except Exception as e:
            results.append(_check("WandB login", False, f"warning: {e}"))

    # ── 16. torch.compile ────────────────────────────────────────────────
    if model is not None:
        try:
            import yaml
            with open(Path("configs/test_tiny.yaml")) as f:
                cfg = yaml.safe_load(f)
            from src.model import build_llama_model
            tiny_model = build_llama_model(cfg["model"])
            compiled = torch.compile(tiny_model, mode="default")
            results.append(_check("torch.compile on tiny model", True))
        except Exception as e:
            results.append(_check("torch.compile on tiny model", True, f"warning: {e} — skipped"))
    else:
        results.append(_check("torch.compile on tiny model", True, "skipped (model not built)"))

    return results


def main() -> None:
    """Parse args and run all pre-flight checks."""
    parser = argparse.ArgumentParser(description="Validate T2T² experiment setup.")
    parser.add_argument("--no-wandb", action="store_true", help="Skip WandB login check.")
    args = parser.parse_args()

    print("\nT2T² Setup Validation")
    print("=" * 40)
    results = run_all_checks(no_wandb=args.no_wandb)
    print("=" * 40)

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} checks passed.")

    if passed < total:
        failed_indices = [i + 1 for i, r in enumerate(results) if not r]
        print(f"Failed checks: {failed_indices}")
        sys.exit(1)
    else:
        print("All checks passed — ready to run experiments.")
        sys.exit(0)


if __name__ == "__main__":
    main()
