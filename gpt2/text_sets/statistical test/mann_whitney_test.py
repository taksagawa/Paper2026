"""
Mann-Whitney U test for sigma_token_per_T and sigma_block_per_T
between causal and noncausal text groups.

Usage:
    Place this script in the same folder as raw_results_fixed_texts.csv, then run:
        python mann_whitney_test.py
    Or specify a CSV path explicitly:
        python mann_whitney_test.py path/to/file.csv
"""

import sys
import os
import csv
import numpy as np
from scipy import stats

DEFAULT_CSV = "raw_results_fixed_texts.csv"


def load_data(path):
    causal_token, causal_block = [], []
    noncausal_token, noncausal_block = [], []

    required_columns = ["category", "sigma_token_per_T", "sigma_block_per_T"]

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV file has no header row.")

        missing = [col for col in required_columns if col not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Missing required column(s): {', '.join(missing)}"
            )

        for line_num, row in enumerate(reader, start=2):
            cat_raw = row["category"]
            cat = cat_raw.strip() if cat_raw is not None else ""

            try:
                st = float(row["sigma_token_per_T"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Line {line_num}: invalid sigma_token_per_T value "
                    f"{row['sigma_token_per_T']!r}"
                )

            try:
                sb = float(row["sigma_block_per_T"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Line {line_num}: invalid sigma_block_per_T value "
                    f"{row['sigma_block_per_T']!r}"
                )

            if cat == "causal":
                causal_token.append(st)
                causal_block.append(sb)
            elif cat in ("noncausal", "non-causal"):
                noncausal_token.append(st)
                noncausal_block.append(sb)
            else:
                raise ValueError(
                    f"Line {line_num}: unknown category {cat!r}. "
                    "Expected 'causal', 'noncausal', or 'non-causal'."
                )

    if len(causal_token) == 0:
        raise ValueError("No causal samples found in the CSV file.")
    if len(noncausal_token) == 0:
        raise ValueError("No noncausal samples found in the CSV file.")

    return (
        np.array(causal_token),
        np.array(noncausal_token),
        np.array(causal_block),
        np.array(noncausal_block),
    )


def rank_biserial(U, n1, n2):
    """Effect size r = 2U/(n1*n2) - 1, positive when x tends to be larger than y."""
    return 2.0 * U / (n1 * n2) - 1.0


def remove_outliers(arr):
    """Remove values outside 1.5×IQR from the box edges (Q1 and Q3)."""
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (arr >= lower) & (arr <= upper)
    return arr[mask]


def run_test(x, y, label):
    U, p = stats.mannwhitneyu(x, y, alternative="two-sided", method="exact")
    n1, n2 = len(x), len(y)
    r = rank_biserial(U, n1, n2)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Causal     (n={n1}):  median = {np.median(x):.4f},  mean = {np.mean(x):.4f},  std = {np.std(x, ddof=1):.4f}")
    print(f"  Noncausal  (n={n2}):  median = {np.median(y):.4f},  mean = {np.mean(y):.4f},  std = {np.std(y, ddof=1):.4f}")
    print(f"  U = {U:.1f},  p = {p:.4e},  rank-biserial r = {r:.4f}")
    return {
        "label": label,
        "n_causal": n1,
        "n_noncausal": n2,
        "median_causal": np.median(x),
        "median_noncausal": np.median(y),
        "mean_causal": np.mean(x),
        "mean_noncausal": np.mean(y),
        "U": U,
        "p": p,
        "rank_biserial_r": r,
    }


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.path.join(os.getcwd(), DEFAULT_CSV)

    if not os.path.isfile(path):
        print(f"Error: {path} not found.")
        print(f"Place {DEFAULT_CSV} in the current directory, or specify the path as an argument.")
        sys.exit(1)

    ct, nt, cb, nb = load_data(path)
    print(f"Loaded {len(ct)} causal and {len(nt)} noncausal samples from {path}")

    results = []
    results.append(run_test(ct, nt, "sigma_token_per_T"))
    results.append(run_test(cb, nb, "sigma_block_per_T"))

    # --- Outlier-removed tests ---
    print(f"\n{'#'*60}")
    print(f"  Repeating tests after removing outliers")
    print(f"  (values outside 1.5×IQR from Q1/Q3)")
    print(f"{'#'*60}")

    for x, y, label_base in [
        (ct, nt, "sigma_token_per_T"),
        (cb, nb, "sigma_block_per_T"),
    ]:
        x_clean = remove_outliers(x)
        y_clean = remove_outliers(y)
        print(f"\n  {label_base}: removed {len(x)-len(x_clean)} causal "
              f"and {len(y)-len(y_clean)} noncausal outlier(s)")
        results.append(run_test(x_clean, y_clean,
                                f"{label_base} (outliers removed)"))

    out_path = os.path.join(os.path.dirname(os.path.abspath(path)), "mann_whitney_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
