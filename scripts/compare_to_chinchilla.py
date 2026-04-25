"""Compare T2T² optimal allocation against Chinchilla-optimal baseline."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


BUDGETS = [1e20, 1e22, 1e24, 1e26]
E = 512  # mean inference tokens per sample


def logistic(x: float) -> float:
    """Inverse logit (sigmoid)."""
    return 1.0 / (1.0 + np.exp(-x))


def predict_accuracy(N: float, D: float, k: float, params: dict) -> float:
    """Predict accuracy using fitted scaling law.

    Args:
        N: Non-embedding parameters.
        D: Training tokens.
        k: Test-time samples.
        params: Dict with A, alpha, B, beta, C, gamma.

    Returns:
        Predicted accuracy in [0, 1].
    """
    logit_acc = (
        params["A"] / (N ** params["alpha"])
        + params["B"] / (D ** params["beta"])
        + params["C"] / (k ** params["gamma"])
    )
    return float(logistic(logit_acc))


def chinchilla_allocation(budget: float) -> tuple[float, float]:
    """Compute Chinchilla-optimal (N, D) for a training-only FLOP budget.

    Chinchilla: N_opt = D_opt = sqrt(C / 6)

    Args:
        budget: Total training FLOP budget.

    Returns:
        Tuple of (N, D).
    """
    val = np.sqrt(budget / 6.0)
    return float(val), float(val)


def bootstrap_p_value(
    acc_t2t2: float,
    acc_chin: float,
    n_bootstrap: int = 1000,
    rng_seed: int = 42,
) -> float:
    """Estimate p-value for the improvement via bootstrap simulation.

    Simulates accuracy samples as Bernoulli draws and asks: how often does
    Chinchilla beat T2T²?

    Args:
        acc_t2t2: Predicted T2T² accuracy.
        acc_chin: Predicted Chinchilla accuracy.
        n_bootstrap: Number of bootstrap iterations.
        rng_seed: Random seed.

    Returns:
        Two-tailed p-value.
    """
    rng = np.random.default_rng(rng_seed)
    n_problems = 500  # representative test-set size

    count = 0
    for _ in range(n_bootstrap):
        sample_t2t2 = rng.binomial(n_problems, acc_t2t2) / n_problems
        sample_chin = rng.binomial(n_problems, acc_chin) / n_problems
        if sample_chin >= sample_t2t2:
            count += 1

    return count / n_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare T2T² vs Chinchilla-optimal at matched compute budgets."
    )
    parser.add_argument(
        "--fit", type=Path, default=Path("results/fit_results.json"),
        help="Fitted scaling law JSON (default: results/fit_results.json).",
    )
    parser.add_argument(
        "--pareto", type=Path, default=Path("results/pareto_optimal.json"),
        help="Pareto-optimal allocation JSON (default: results/pareto_optimal.json).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/chinchilla_comparison.json"),
        help="Output JSON (default: results/chinchilla_comparison.json).",
    )
    args = parser.parse_args()

    for path in (args.fit, args.pareto):
        if not path.exists():
            print(f"Error: {path} not found.", file=sys.stderr)
            sys.exit(1)

    with open(args.fit) as f:
        fit = json.load(f)
    with open(args.pareto) as f:
        pareto = json.load(f)

    params = fit.get("pass@1")
    if not params:
        print("Error: 'pass@1' params not found in fit_results.json.", file=sys.stderr)
        sys.exit(1)

    comparisons = []

    for budget in BUDGETS:
        key = f"budget_{_fmt_sci(budget)}"
        t2t2_entry = pareto.get(key, {})
        if not t2t2_entry or t2t2_entry.get("predicted_acc", -1) < 0:
            print(f"Warning: no T2T² allocation for budget {_fmt_sci(budget)} — skipping.")
            continue

        N_chin, D_chin = chinchilla_allocation(budget)
        k_chin = 1
        acc_chin = predict_accuracy(N_chin, D_chin, float(k_chin), params)

        N_t2t2 = t2t2_entry["N"]
        D_t2t2 = t2t2_entry["D"]
        k_t2t2 = t2t2_entry["k"]
        acc_t2t2 = t2t2_entry["predicted_acc"]

        improvement_abs = acc_t2t2 - acc_chin
        improvement_pct = (improvement_abs / max(acc_chin, 1e-9)) * 100.0
        p_value = bootstrap_p_value(acc_t2t2, acc_chin)
        significant = p_value < 0.05

        entry = {
            "budget": budget,
            "chinchilla": {
                "N": float(N_chin), "D": float(D_chin), "k": k_chin,
                "accuracy": acc_chin,
            },
            "t2t2": {
                "N": N_t2t2, "D": D_t2t2, "k": k_t2t2,
                "accuracy": acc_t2t2,
            },
            "improvement_abs": float(improvement_abs),
            "improvement_pct": float(improvement_pct),
            "p_value": float(p_value),
            "significant": significant,
        }
        comparisons.append(entry)

        sig_str = "YES" if significant else "NO"
        print(
            f"Budget {_fmt_sci(budget)}: T2T² improves over Chinchilla by "
            f"{improvement_pct:+.1f}% (p={p_value:.3f}, significant: {sig_str})"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(comparisons, f, indent=2)

    print(f"\nSaved to {args.output}")


def _fmt_sci(x: float) -> str:
    """Format a float in scientific notation."""
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    return f"{x/10**exp:.0f}e{exp}"


if __name__ == "__main__":
    main()
