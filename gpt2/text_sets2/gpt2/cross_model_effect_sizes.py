"""Cross-generator effect-size analysis for the fixed-text GPT-2 experiment.

This script reads the five model-tagged raw-result CSV files, validates their
structure, reproduces the two-sided asymptotic Mann-Whitney U tests used in the
paper, and creates:

1. a two-panel cross-generator plot of rank-biserial correlations with
   bootstrap 95% confidence intervals;
2. a primary-analysis CSV with raw and Holm-adjusted p-values;
3. a sensitivity-analysis CSV after removing texts shared by all five corpora;
4. a CSV listing the shared texts that were removed; and
5. a JSON file recording inputs and analysis settings.

The primary analysis always uses all observations. No outlier removal is
performed. Positive rank-biserial correlation means that the causal group
tends to have larger values than the non-causal group.

Example:
    python cross_model_effect_sizes.py

Statistics-only verification when Matplotlib is unavailable:
    python cross_model_effect_sizes.py --skip-figure --bootstrap-reps 1000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


MODEL_SPECS = [
    ("Opus46", "Claude Opus 4.6"),
    ("Fable5", "Claude Fable 5"),
    ("Gemini31", "Gemini 3.1 Pro"),
    ("GPT54", "GPT-5.4 Pro"),
    ("GPT56", "GPT-5.6 Pro"),
]

METRICS = [
    ("sigma_token_per_T", "Token reversal", "#7B2CBF"),
    ("sigma_block_per_T", "Block reversal", "#159F80"),
]

REQUIRED_COLUMNS = {
    "category",
    "text",
    "T",
    "n_blocks",
    "sigma_token_per_T",
    "sigma_block_per_T",
}

RESULT_FIELDS = [
    "analysis",
    "model_tag",
    "input_text_generator",
    "metric",
    "n_causal",
    "n_noncausal",
    "mean_causal",
    "mean_noncausal",
    "median_causal",
    "median_noncausal",
    "U_causal",
    "p_raw",
    "p_holm",
    "rank_biserial_r",
    "r_ci_low",
    "r_ci_high",
    "ci_level",
    "bootstrap_reps",
    "bootstrap_seed",
    "direction",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Create a five-generator effect-size plot and statistical tables "
            "from raw fixed-text GPT-2 results."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory containing raw_results_fixed_texts_<TAG>.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory for the figure and output tables.",
    )
    parser.add_argument(
        "--expected-n",
        type=int,
        default=100,
        help="Expected number of texts in each category and corpus.",
    )
    parser.add_argument(
        "--expected-blocks",
        type=int,
        default=4,
        help="Expected number of sentence blocks in every text; use 0 to disable.",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=10_000,
        help="Number of stratified bootstrap replicates for the primary CIs.",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Bootstrap confidence level (default: 0.95).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--skip-figure",
        action="store_true",
        help="Write statistical outputs without importing Matplotlib or plotting.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively in addition to saving it.",
    )
    args = parser.parse_args(argv)

    if args.expected_n <= 0:
        parser.error("--expected-n must be positive.")
    if args.expected_blocks < 0:
        parser.error("--expected-blocks must be non-negative.")
    if args.bootstrap_reps <= 0:
        parser.error("--bootstrap-reps must be positive.")
    if not 0.0 < args.ci_level < 1.0:
        parser.error("--ci-level must lie strictly between 0 and 1.")
    return args


def canonical_category(value, *, path, line_number):
    category = (value or "").strip().lower()
    if category == "causal":
        return "causal"
    if category in {"noncausal", "non-causal"}:
        return "noncausal"
    raise ValueError(
        f"{path}, line {line_number}: unknown category {value!r}; "
        "expected 'causal', 'noncausal', or 'non-causal'."
    )


def parse_finite_float(value, *, column, path, line_number):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}, line {line_number}: invalid {column} value {value!r}."
        ) from exc
    if not math.isfinite(result):
        raise ValueError(
            f"{path}, line {line_number}: non-finite {column} value {value!r}."
        )
    return result


def parse_integer(value, *, column, path, line_number):
    result = parse_finite_float(
        value, column=column, path=path, line_number=line_number
    )
    if not result.is_integer():
        raise ValueError(
            f"{path}, line {line_number}: {column} must be an integer, got {value!r}."
        )
    return int(result)


def load_dataset(path, *, expected_n, expected_blocks):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header row.")
        missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")

        records = []
        for line_number, row in enumerate(reader, start=2):
            category = canonical_category(
                row.get("category"), path=path, line_number=line_number
            )
            text = (row.get("text") or "").strip()
            if not text:
                raise ValueError(f"{path}, line {line_number}: text must be nonempty.")
            token_count = parse_integer(
                row.get("T"), column="T", path=path, line_number=line_number
            )
            block_count = parse_integer(
                row.get("n_blocks"),
                column="n_blocks",
                path=path,
                line_number=line_number,
            )
            if token_count <= 0:
                raise ValueError(f"{path}, line {line_number}: T must be positive.")
            if expected_blocks and block_count != expected_blocks:
                raise ValueError(
                    f"{path}, line {line_number}: expected {expected_blocks} blocks, "
                    f"got {block_count}."
                )

            record = {
                "category": category,
                "text": text,
                "T": token_count,
                "n_blocks": block_count,
            }
            for metric, _, _ in METRICS:
                record[metric] = parse_finite_float(
                    row.get(metric),
                    column=metric,
                    path=path,
                    line_number=line_number,
                )
            records.append(record)

    counts = {
        category: sum(record["category"] == category for record in records)
        for category in ("causal", "noncausal")
    }
    for category, count in counts.items():
        if count != expected_n:
            raise ValueError(
                f"{path}: expected {expected_n} {category} texts, got {count}."
            )

    texts = [record["text"] for record in records]
    duplicate_count = len(texts) - len(set(texts))
    if duplicate_count:
        raise ValueError(f"{path}: found {duplicate_count} duplicate text row(s).")

    return records


def values_for(records, category, metric, excluded_texts=frozenset()):
    return np.asarray(
        [
            record[metric]
            for record in records
            if record["category"] == category
            and record["text"] not in excluded_texts
        ],
        dtype=np.float64,
    )


def mann_whitney_asymptotic(x, y):
    """Two-sided asymptotic Mann-Whitney test with tie/continuity correction.

    Returns U for the first sample x, matching scipy.stats.mannwhitneyu(x, y,
    alternative='two-sided', method='asymptotic', use_continuity=True).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        raise ValueError("Mann-Whitney test requires two nonempty samples.")

    combined = np.concatenate([x, y])
    order = np.argsort(combined, kind="mergesort")
    sorted_values = combined[order]
    ranks = np.empty(len(combined), dtype=np.float64)
    tie_sizes = []

    start = 0
    while start < len(sorted_values):
        stop = start + 1
        while stop < len(sorted_values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        tie_sizes.append(stop - start)
        start = stop

    rank_sum_x = float(ranks[:n1].sum())
    u1 = rank_sum_x - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    mean_u = n1 * n2 / 2.0
    total_n = n1 + n2
    tie_term = sum(size**3 - size for size in tie_sizes)
    variance = n1 * n2 / 12.0 * (
        (total_n + 1) - tie_term / (total_n * (total_n - 1))
    )

    if variance <= 0.0:
        p_value = 1.0
    else:
        z = (max(u1, u2) - mean_u - 0.5) / math.sqrt(variance)
        p_value = min(1.0, math.erfc(z / math.sqrt(2.0)))
    return u1, p_value


def rank_biserial_from_u(u_value, n1, n2):
    return 2.0 * u_value / (n1 * n2) - 1.0


def generate_bootstrap_counts(n1, n2, repetitions, rng):
    probabilities_x = np.full(n1, 1.0 / n1, dtype=np.float64)
    probabilities_y = np.full(n2, 1.0 / n2, dtype=np.float64)
    counts_x = rng.multinomial(n1, probabilities_x, size=repetitions).astype(
        np.float64, copy=False
    )
    counts_y = rng.multinomial(n2, probabilities_y, size=repetitions).astype(
        np.float64, copy=False
    )
    return counts_x, counts_y


def bootstrap_rank_biserial_ci(x, y, counts_x, counts_y, ci_level):
    """Percentile bootstrap CI using exact resampling multiplicities.

    The comparison matrix contains +1 for x > y, -1 for x < y, and 0 for
    ties. Weighting it by bootstrap multinomial counts exactly reproduces the
    rank-biserial correlation of each resampled pair of groups.
    """
    comparison = np.sign(x[:, None] - y[None, :]).astype(np.float64, copy=False)
    weighted = counts_x @ comparison
    bootstrap_r = np.einsum(
        "bi,bi->b", weighted, counts_y, optimize=True
    ) / (len(x) * len(y))
    alpha = 1.0 - ci_level
    low, high = np.quantile(bootstrap_r, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values, kind="mergesort")
    sorted_p = p_values[order]
    factors = len(sorted_p) - np.arange(len(sorted_p))
    adjusted_sorted = np.maximum.accumulate(sorted_p * factors)
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def direction_label(effect_size):
    if effect_size > 0:
        return "causal > non-causal"
    if effect_size < 0:
        return "causal < non-causal"
    return "no directional difference"


def compute_result(
    *,
    analysis,
    model_tag,
    model_label,
    metric,
    x,
    y,
    ci_level,
    bootstrap_reps,
    bootstrap_seed,
    bootstrap_counts=None,
):
    u_value, p_value = mann_whitney_asymptotic(x, y)
    effect_size = rank_biserial_from_u(u_value, len(x), len(y))
    if bootstrap_counts is None:
        ci_low = ""
        ci_high = ""
        effective_bootstrap_reps = 0
    else:
        counts_x, counts_y = bootstrap_counts
        ci_low, ci_high = bootstrap_rank_biserial_ci(
            x, y, counts_x, counts_y, ci_level
        )
        effective_bootstrap_reps = bootstrap_reps

    return {
        "analysis": analysis,
        "model_tag": model_tag,
        "input_text_generator": model_label,
        "metric": metric,
        "n_causal": len(x),
        "n_noncausal": len(y),
        "mean_causal": float(np.mean(x)),
        "mean_noncausal": float(np.mean(y)),
        "median_causal": float(np.median(x)),
        "median_noncausal": float(np.median(y)),
        "U_causal": u_value,
        "p_raw": p_value,
        "p_holm": "",
        "rank_biserial_r": effect_size,
        "r_ci_low": ci_low,
        "r_ci_high": ci_high,
        "ci_level": ci_level if bootstrap_counts is not None else "",
        "bootstrap_reps": effective_bootstrap_reps,
        "bootstrap_seed": bootstrap_seed if bootstrap_counts is not None else "",
        "direction": direction_label(effect_size),
    }


def apply_holm(rows):
    adjusted = holm_adjust([row["p_raw"] for row in rows])
    for row, p_adjusted in zip(rows, adjusted):
        row["p_holm"] = float(p_adjusted)


def find_texts_shared_by_all(datasets):
    shared = {}
    for category in ("causal", "noncausal"):
        per_model_sets = []
        for model_tag, _ in MODEL_SPECS:
            per_model_sets.append(
                {
                    record["text"]
                    for record in datasets[model_tag]
                    if record["category"] == category
                }
            )
        shared[category] = set.intersection(*per_model_sets)
    return shared


def write_results_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_shared_texts_csv(path, shared_texts):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["category", "text", "model_count"]
        )
        writer.writeheader()
        for category in ("causal", "noncausal"):
            for text in sorted(shared_texts[category]):
                writer.writerow(
                    {
                        "category": category,
                        "text": text,
                        "model_count": len(MODEL_SPECS),
                    }
                )


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_effect_sizes(rows, output_dir, *, show):
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required to create the figure. Install Matplotlib "
            "or rerun with --skip-figure to verify statistical outputs only."
        ) from exc

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 2.4), sharey=True)
    y_positions = np.arange(len(MODEL_SPECS), dtype=float)
    model_labels = [label for _, label in MODEL_SPECS]

    for panel_index, (metric, title, color) in enumerate(METRICS):
        ax = axes[panel_index]
        metric_rows = {
            row["model_tag"]: row for row in rows if row["metric"] == metric
        }
        ordered = [metric_rows[tag] for tag, _ in MODEL_SPECS]
        effects = np.asarray(
            [row["rank_biserial_r"] for row in ordered], dtype=float
        )
        lows = np.asarray([row["r_ci_low"] for row in ordered], dtype=float)
        highs = np.asarray([row["r_ci_high"] for row in ordered], dtype=float)

        ax.axvline(0.0, color="#4B5563", linestyle="--", linewidth=1.1, zorder=1)
        ax.hlines(y_positions, lows, highs, color=color, linewidth=2.2, zorder=2)
        cap_height = 0.09
        ax.vlines(
            lows,
            y_positions - cap_height,
            y_positions + cap_height,
            color=color,
            linewidth=1.5,
            zorder=2,
        )
        ax.vlines(
            highs,
            y_positions - cap_height,
            y_positions + cap_height,
            color=color,
            linewidth=1.5,
            zorder=2,
        )
        marker = "o" if metric == "sigma_token_per_T" else "D"
        ax.scatter(
            effects,
            y_positions,
            s=65,
            marker=marker,
            facecolor=color,
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )

        ax.set_xlim(-1.0, 1.0)
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_xlabel(
            "Rank-biserial correlation $r$\n(positive: causal > non-causal)",
            fontsize=11,
        )
        ax.set_title(f"({chr(97 + panel_index)}) {title}", fontsize=13)
        ax.grid(axis="x", color="#D1D5DB", linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=10)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(model_labels, fontsize=10)
    axes[0].invert_yaxis()

    fig.subplots_adjust(left=0.23, right=0.985, bottom=0.20, top=0.90, wspace=0.16)
    png_path = output_dir / "cross_model_effect_sizes.png"
    pdf_path = output_dir / "cross_model_effect_sizes.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return png_path, pdf_path


def main(argv=None):
    args = parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {}
    input_paths = {}
    for model_tag, _ in MODEL_SPECS:
        path = input_dir / f"raw_results_fixed_texts_{model_tag}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required raw-result file not found: {path}")
        datasets[model_tag] = load_dataset(
            path,
            expected_n=args.expected_n,
            expected_blocks=args.expected_blocks,
        )
        input_paths[model_tag] = path
        print(f"Validated {path.name}: {len(datasets[model_tag])} rows")

    shared_texts = find_texts_shared_by_all(datasets)
    print(
        "Texts shared by all corpora: "
        f"{len(shared_texts['causal'])} causal, "
        f"{len(shared_texts['noncausal'])} non-causal"
    )

    primary_rows = []
    for model_index, (model_tag, model_label) in enumerate(MODEL_SPECS):
        records = datasets[model_tag]
        example_metric = METRICS[0][0]
        n_causal = len(values_for(records, "causal", example_metric))
        n_noncausal = len(values_for(records, "noncausal", example_metric))
        model_seed_sequence = np.random.SeedSequence([args.seed, model_index])
        rng = np.random.default_rng(model_seed_sequence)
        bootstrap_counts = generate_bootstrap_counts(
            n_causal, n_noncausal, args.bootstrap_reps, rng
        )

        for metric, _, _ in METRICS:
            x = values_for(records, "causal", metric)
            y = values_for(records, "noncausal", metric)
            primary_rows.append(
                compute_result(
                    analysis="primary_all_observations",
                    model_tag=model_tag,
                    model_label=model_label,
                    metric=metric,
                    x=x,
                    y=y,
                    ci_level=args.ci_level,
                    bootstrap_reps=args.bootstrap_reps,
                    bootstrap_seed=args.seed,
                    bootstrap_counts=bootstrap_counts,
                )
            )
        del bootstrap_counts
    apply_holm(primary_rows)

    sensitivity_rows = []
    for model_tag, model_label in MODEL_SPECS:
        records = datasets[model_tag]
        for metric, _, _ in METRICS:
            x = values_for(
                records, "causal", metric, shared_texts["causal"]
            )
            y = values_for(
                records, "noncausal", metric, shared_texts["noncausal"]
            )
            sensitivity_rows.append(
                compute_result(
                    analysis="sensitivity_shared_texts_removed",
                    model_tag=model_tag,
                    model_label=model_label,
                    metric=metric,
                    x=x,
                    y=y,
                    ci_level=args.ci_level,
                    bootstrap_reps=0,
                    bootstrap_seed=args.seed,
                    bootstrap_counts=None,
                )
            )
    apply_holm(sensitivity_rows)

    primary_path = output_dir / "cross_model_statistics.csv"
    sensitivity_path = (
        output_dir / "cross_model_sensitivity_shared_texts_removed.csv"
    )
    shared_path = output_dir / "cross_model_shared_texts_removed.csv"
    metadata_path = output_dir / "cross_model_analysis_metadata.json"
    write_results_csv(primary_path, primary_rows)
    write_results_csv(sensitivity_path, sensitivity_rows)
    write_shared_texts_csv(shared_path, shared_texts)

    metadata = {
        "analysis": "Cross-generator fixed-text GPT-2 entropy production",
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "model_order": [
            {"tag": model_tag, "label": model_label}
            for model_tag, model_label in MODEL_SPECS
        ],
        "input_files": {
            model_tag: {
                "path": str(input_paths[model_tag]),
                "sha256": file_sha256(input_paths[model_tag]),
            }
            for model_tag, _ in MODEL_SPECS
        },
        "primary_analysis": {
            "n_per_category": args.expected_n,
            "outlier_removal": False,
            "test": "two-sided asymptotic Mann-Whitney U",
            "continuity_correction": True,
            "effect_size": "rank-biserial correlation; positive means causal > non-causal",
            "multiple_testing": "Holm adjustment across 10 primary tests",
            "bootstrap_repetitions": args.bootstrap_reps,
            "bootstrap_seed": args.seed,
            "confidence_level": args.ci_level,
            "bootstrap_scheme": "independent within-category resampling; percentile interval",
        },
        "sensitivity_analysis": {
            "description": "Remove texts shared by all five corpora within each category",
            "shared_causal_text_count": len(shared_texts["causal"]),
            "shared_noncausal_text_count": len(shared_texts["noncausal"]),
            "multiple_testing": "Holm adjustment across 10 sensitivity tests",
            "confidence_intervals": False,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    figure_paths = []
    if not args.skip_figure:
        figure_paths = list(plot_effect_sizes(primary_rows, output_dir, show=args.show))

    print(f"Saved primary statistics: {primary_path}")
    print(f"Saved sensitivity statistics: {sensitivity_path}")
    print(f"Saved shared-text list: {shared_path}")
    print(f"Saved analysis metadata: {metadata_path}")
    for figure_path in figure_paths:
        print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()
