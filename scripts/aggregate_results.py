"""Aggregate all experiment result JSONs into a single CSV."""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path


COLUMNS = [
    "run_id", "model_size", "total_params", "non_embed_params",
    "D_tokens", "k_values", "pass@1", "pass@4", "pass@8", "pass@16",
    "maj@1", "maj@4", "maj@8", "maj@16",
    "best_val_loss", "compute_flops", "train_flops", "infer_flops",
    "seed", "timestamp", "git_commit",
]


def aggregate(results_dir: Path, output_path: Path) -> int:
    """Load all result JSONs and write a unified CSV.

    Args:
        results_dir: Directory containing result JSON files.
        output_path: Path for the output CSV file.

    Returns:
        Number of experiments successfully aggregated.
    """
    json_files = sorted(glob.glob(str(results_dir / "*.json")))
    # Exclude nested outputs that aren't experiment results
    json_files = [f for f in json_files if Path(f).name != "aggregate.json"]

    if not json_files:
        print("No results found.")
        return 0

    rows = []
    for path in json_files:
        try:
            with open(path) as f:
                data = json.load(f)
            row = {col: data.get(col, "") for col in COLUMNS}
            rows.append(row)
        except Exception as e:
            print(f"Warning: skipping {path} — {e}", file=sys.stderr)

    if not rows:
        print("No valid results found.")
        return 0

    rows.sort(key=lambda r: r.get("run_id", ""))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Aggregated {len(rows)} experiments into {output_path}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate T2T² result JSONs into a CSV.")
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Directory containing result JSON files (default: results/).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/aggregate.csv"),
        help="Output CSV path (default: results/aggregate.csv).",
    )
    args = parser.parse_args()
    aggregate(args.results_dir, args.output)


if __name__ == "__main__":
    main()
