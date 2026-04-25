"""Fit unified scaling law: logit(acc) = A/N^α + B/D^β + C/k^γ."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def logit(p: np.ndarray) -> np.ndarray:
    """Compute logit, clamping p away from 0 and 1 to avoid inf."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def scaling_law(X: np.ndarray, A: float, alpha: float, B: float, beta: float, C: float, gamma: float) -> np.ndarray:
    """Scaling law model: logit(acc) = A/N^α + B/D^β + C/k^γ.

    Args:
        X: Array of shape (3, n) with rows [N, D, k].
        A, alpha, B, beta, C, gamma: Law parameters.

    Returns:
        Predicted logit(accuracy).
    """
    N, D, k = X[0], X[1], X[2]
    return A / (N ** alpha) + B / (D ** beta) + C / (k ** gamma)


def fit_metric(df: pd.DataFrame, metric: str) -> dict:
    """Fit the scaling law for a single accuracy metric.

    Args:
        df: Aggregated results DataFrame.
        metric: Column name to fit (e.g. 'pass@1', 'maj@8').

    Returns:
        Dict of fitted parameters and diagnostics.
    """
    sub = df[["non_embed_params", "D_tokens", metric]].dropna()
    # k is encoded in the metric name (e.g. pass@8 → k=8)
    k_val = int(metric.split("@")[1])

    N = sub["non_embed_params"].values.astype(float)
    D = sub["D_tokens"].values.astype(float)
    k = np.full_like(N, float(k_val))
    y = logit(sub[metric].values.astype(float))

    X = np.vstack([N, D, k])
    p0 = [1.0, 0.3, 1.0, 0.4, 1.0, 0.3]
    bounds = ([0, 0, 0, 0, 0, 0], [np.inf, 2, np.inf, 2, np.inf, 2])

    try:
        popt, _ = curve_fit(scaling_law, X, y, p0=p0, bounds=bounds, maxfev=10000)
    except RuntimeError as e:
        print(f"Warning: curve_fit failed for {metric}: {e}", file=sys.stderr)
        return {}

    A, alpha, B, beta, C, gamma = popt

    y_pred = scaling_law(X, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Bootstrap confidence intervals (100 samples)
    n = len(y)
    boot_params = []
    rng = np.random.default_rng(42)
    for _ in range(100):
        idx = rng.integers(0, n, size=n)
        try:
            bp, _ = curve_fit(
                scaling_law, X[:, idx], y[idx],
                p0=popt, bounds=bounds, maxfev=5000,
            )
            boot_params.append(bp)
        except RuntimeError:
            pass

    boot_params = np.array(boot_params) if boot_params else np.zeros((1, 6))
    ci = np.percentile(boot_params, [2.5, 97.5], axis=0)

    return {
        "A": float(A), "alpha": float(alpha),
        "B": float(B), "beta": float(beta),
        "C": float(C), "gamma": float(gamma),
        "r_squared": float(r2),
        "ci_A": ci[:, 0].tolist(),
        "ci_alpha": ci[:, 1].tolist(),
        "ci_B": ci[:, 2].tolist(),
        "ci_beta": ci[:, 3].tolist(),
        "ci_C": ci[:, 4].tolist(),
        "ci_gamma": ci[:, 5].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit T2T² scaling law from aggregated CSV.")
    parser.add_argument(
        "--input", type=Path, default=Path("results/aggregate.csv"),
        help="Aggregated CSV (default: results/aggregate.csv).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/fit_results.json"),
        help="Output JSON (default: results/fit_results.json).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found. Run aggregate_results.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)

    numeric_cols = ["non_embed_params", "D_tokens", "pass@1", "pass@8", "maj@8"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.empty or len(df) < 4:
        print(f"Error: not enough rows in {args.input} to fit (need ≥ 4).", file=sys.stderr)
        sys.exit(1)

    results = {
        "formula": "logit(acc) = A/N^α + B/D^β + C/k^γ",
        "pass@1": fit_metric(df, "pass@1"),
        "maj@8": fit_metric(df, "maj@8"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    for metric in ("pass@1", "maj@8"):
        r = results.get(metric, {})
        if r:
            print(
                f"Scaling law fitted. R² = {r.get('r_squared', float('nan')):.2f} ({metric}). "
                f"α={r.get('alpha', float('nan')):.2f}, "
                f"β={r.get('beta', float('nan')):.2f}, "
                f"γ={r.get('gamma', float('nan')):.2f}"
            )

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
