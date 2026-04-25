"""Find optimal (N, D, k) allocation for fixed compute budgets using the fitted scaling law."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# Default grid values
N_GRID = [3_200_000, 6_400_000, 12_800_000, 25_600_000, 50_000_000, 100_000_000]
K_GRID = [1, 2, 4, 8, 16, 32]
BUDGETS = [1e20, 1e22, 1e24, 1e26]
E = 512  # mean tokens generated per inference sample


def logistic(x: float) -> float:
    """Inverse logit (sigmoid)."""
    return 1.0 / (1.0 + np.exp(-x))


def predict_accuracy(N: float, D: float, k: float, params: dict) -> float:
    """Predict accuracy using the fitted scaling law.

    Args:
        N: Non-embedding parameter count.
        D: Training token count.
        k: Number of test-time samples.
        params: Dict with keys A, alpha, B, beta, C, gamma.

    Returns:
        Predicted accuracy in [0, 1].
    """
    A, alpha = params["A"], params["alpha"]
    B, beta = params["B"], params["beta"]
    C, gamma = params["C"], params["gamma"]
    logit_acc = A / (N ** alpha) + B / (D ** beta) + C / (k ** gamma)
    return float(logistic(logit_acc))


def find_optimal(budget: float, params: dict) -> dict:
    """Grid-search optimal (N, D, k) for a given compute budget.

    Training FLOPs: 6 * N * D
    Inference FLOPs: 2 * N * k * E
    Total: 6*N*D + 2*N*k*E ≤ budget

    Args:
        budget: Total FLOP budget.
        params: Fitted scaling law parameters (for pass@1 or maj@8).

    Returns:
        Dict with N, D, k, predicted_acc, compute.
    """
    best = {"predicted_acc": -1.0}

    for N in N_GRID:
        for k in K_GRID:
            # Solve for D: 6*N*D + 2*N*k*E = budget  →  D = (budget - 2*N*k*E) / (6*N)
            infer_flops = 2 * N * k * E
            if infer_flops >= budget:
                continue
            D = (budget - infer_flops) / (6 * N)
            if D < 1e6:  # less than 1M tokens is not meaningful
                continue

            acc = predict_accuracy(float(N), float(D), float(k), params)
            total_compute = 6 * N * D + infer_flops

            if acc > best["predicted_acc"]:
                best = {
                    "N": int(N),
                    "D": float(D),
                    "k": int(k),
                    "predicted_acc": float(acc),
                    "compute": float(total_compute),
                }

    return best


def _fmt_sci(x: float) -> str:
    """Format a float in scientific notation for table printing."""
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    return f"{x/10**exp:.1f}e{exp}"


def _fmt_params(n: int) -> str:
    """Format parameter count as e.g. '6.4M'."""
    return f"{n/1e6:.1f}M"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find Pareto-optimal (N, D, k) for fixed compute budgets."
    )
    parser.add_argument(
        "--fit", type=Path, default=Path("results/fit_results.json"),
        help="Fitted scaling law JSON (default: results/fit_results.json).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/pareto_optimal.json"),
        help="Output JSON (default: results/pareto_optimal.json).",
    )
    args = parser.parse_args()

    if not args.fit.exists():
        print(f"Error: {args.fit} not found. Run fit_scaling_law.py first.", file=sys.stderr)
        sys.exit(1)

    with open(args.fit) as f:
        fit = json.load(f)

    # Use pass@1 params as the primary metric for optimization
    params = fit.get("pass@1")
    if not params:
        print("Error: 'pass@1' fit not found in fit_results.json.", file=sys.stderr)
        sys.exit(1)

    output: dict = {}

    header = f"{'Budget':<10} {'N*':<10} {'D*':<14} {'k*':<6} {'Pred Acc':<12} {'Compute':<12}"
    print(header)
    print("-" * len(header))

    for budget in BUDGETS:
        result = find_optimal(budget, params)
        key = f"budget_{_fmt_sci(budget)}"
        output[key] = result

        if result.get("predicted_acc", -1) < 0:
            print(f"{_fmt_sci(budget):<10} — no feasible allocation found")
        else:
            print(
                f"{_fmt_sci(budget):<10} "
                f"{_fmt_params(result['N']):<10} "
                f"{_fmt_sci(result['D']):<14} "
                f"{result['k']:<6} "
                f"{result['predicted_acc']:<12.4f} "
                f"{_fmt_sci(result['compute']):<12}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
