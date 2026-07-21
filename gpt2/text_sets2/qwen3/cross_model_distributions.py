"""Create a compact cross-generator distribution figure for Appendix B4, B5.

The figure combines all five fixed-text corpora in two panels.  Within each
input-text generator, the causal and non-causal distributions are shown as
box plots with all individual observations overlaid.  Token- and block-level
quantities use separate panels but a common scale across generators within
each panel.

Example:
    python cross_model_distributions.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

from cross_model_effect_sizes import MODEL_SPECS, load_dataset, values_for


METRIC_SPECS = [
    ("sigma_token_per_T", "(a) Token reversal", r"$\sigma_{\mathrm{token}}/T$"),
    ("sigma_block_per_T", "(b) Block reversal", r"$\sigma_{\mathrm{block}}/T$"),
]

DISPLAY_LABELS = {
    "Opus46": "Claude\nOpus 4.6",
    "Fable5": "Claude\nFable 5",
    "Gemini31": "Gemini\n3.1 Pro",
    "GPT54": "GPT-5.4\nPro",
    "GPT56": "GPT-5.6\nPro",
}

CATEGORY_SPECS = [
    ("causal", "Causal", "#E76F61", -0.19),
    ("noncausal", "Non-causal", "#5DADE2", 0.19),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Create a two-panel distribution figure containing all five "
            "input-text generators."
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
        help="Directory for cross_model_distributions.pdf and .png.",
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
        "--jitter-seed",
        type=int,
        default=314159,
        help="Random seed used only for deterministic horizontal point jitter.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution of the PNG output.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving it.",
    )
    args = parser.parse_args(argv)

    if args.expected_n <= 0:
        parser.error("--expected-n must be positive.")
    if args.expected_blocks < 0:
        parser.error("--expected-blocks must be non-negative.")
    if args.dpi <= 0:
        parser.error("--dpi must be positive.")
    return args


def load_all_datasets(input_dir, *, expected_n, expected_blocks):
    datasets = {}
    for model_tag, _ in MODEL_SPECS:
        path = input_dir / f"raw_results_fixed_texts_{model_tag}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required raw-result file not found: {path}")
        datasets[model_tag] = load_dataset(
            path,
            expected_n=expected_n,
            expected_blocks=expected_blocks,
        )
        print(f"Validated {path.name}: {len(datasets[model_tag])} rows")
    return datasets


def set_common_y_limits(ax, metric, values):
    all_values = np.concatenate(values)
    if metric == "sigma_token_per_T":
        upper = math.ceil((float(np.max(all_values)) + 0.05) * 10.0) / 10.0
        ax.set_ylim(0.0, upper)
    else:
        limit = math.ceil(float(np.max(np.abs(all_values))) * 1.08 * 20.0) / 20.0
        ax.set_ylim(-limit, limit)


def plot_distributions(datasets, output_dir, *, jitter_seed, dpi, show):
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required to create the figure. Install Matplotlib "
            "and rerun the script."
        ) from exc

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.1))
    model_positions = np.arange(len(MODEL_SPECS), dtype=float)

    for panel_index, (metric, title, ylabel) in enumerate(METRIC_SPECS):
        ax = axes[panel_index]
        panel_values = []

        for model_index, (model_tag, _) in enumerate(MODEL_SPECS):
            records = datasets[model_tag]
            for category_index, (category, _, color, offset) in enumerate(
                CATEGORY_SPECS
            ):
                values = values_for(records, category, metric)
                panel_values.append(values)
                position = model_positions[model_index] + offset

                box = ax.boxplot(
                    [values],
                    positions=[position],
                    widths=0.29,
                    patch_artist=True,
                    showfliers=False,
                    whis=1.5,
                    boxprops={
                        "facecolor": color,
                        "edgecolor": "#1F2937",
                        "linewidth": 1.0,
                        "alpha": 0.62,
                    },
                    medianprops={"color": "#111827", "linewidth": 1.6},
                    whiskerprops={"color": "#374151", "linewidth": 1.0},
                    capprops={"color": "#374151", "linewidth": 1.0},
                    zorder=2,
                )
                # Keep a reference so Matplotlib does not consider the artists unused.
                del box

                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [jitter_seed, panel_index, model_index, category_index]
                    )
                )
                jitter = rng.uniform(-0.105, 0.105, size=len(values))
                ax.scatter(
                    np.full(len(values), position) + jitter,
                    values,
                    s=11,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.38,
                    zorder=3,
                )
                ax.scatter(
                    [position],
                    [float(np.mean(values))],
                    marker="D",
                    s=28,
                    facecolor="#111827",
                    edgecolor="#111827",
                    linewidth=0.5,
                    zorder=4,
                )

        set_common_y_limits(ax, metric, panel_values)
        ax.axhline(0.0, color="#6B7280", linestyle="--", linewidth=0.9, zorder=1)
        ax.set_xlim(-0.58, len(MODEL_SPECS) - 0.42)
        ax.set_xticks(model_positions)
        ax.set_xticklabels(
            [DISPLAY_LABELS[tag] for tag, _ in MODEL_SPECS], fontsize=9
        )
        ax.set_xlabel("Input-text generator", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, pad=8)
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    legend_handles = [
        Patch(
            facecolor=CATEGORY_SPECS[0][2],
            edgecolor="#1F2937",
            alpha=0.62,
            label="Causal",
        ),
        Patch(
            facecolor=CATEGORY_SPECS[1][2],
            edgecolor="#1F2937",
            alpha=0.62,
            label="Non-causal",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor="#111827",
            markeredgecolor="#111827",
            markersize=5,
            label="Mean",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=10,
        columnspacing=1.8,
        handletextpad=0.6,
    )

    fig.subplots_adjust(
        left=0.075, right=0.992, bottom=0.22, top=0.80, wspace=0.18
    )
    png_path = output_dir / "cross_model_distributions.png"
    pdf_path = output_dir / "cross_model_distributions.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
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

    datasets = load_all_datasets(
        input_dir,
        expected_n=args.expected_n,
        expected_blocks=args.expected_blocks,
    )
    png_path, pdf_path = plot_distributions(
        datasets,
        output_dir,
        jitter_seed=args.jitter_seed,
        dpi=args.dpi,
        show=args.show,
    )
    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


if __name__ == "__main__":
    main()
