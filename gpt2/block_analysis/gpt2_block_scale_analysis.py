#!/usr/bin/env python3
"""Block-scale entropy-production analysis for the GPT-2 Figure 3 samples.

This script does not generate new text. It downloads the saved Monte Carlo
samples from GitHub, reconstructs the sequences used for the block-reversal
part of Figure 3, and evaluates two families of reorderings:

1. Sentence grouping: group k consecutive sentences into a superblock and
   reverse the superblock order.
2. Fixed-token grouping: group l consecutive tokens into a block and reverse
   the block order.

Both analyses use the same valid, punctuation-terminated sequences of length
T' so that their per-token entropy productions are directly comparable.

Example
-------
python gpt2_block_scale_analysis.py --device cuda --batch-size 16

Required packages
-----------------
torch, transformers, numpy, matplotlib
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer


SCRIPT_VERSION = "1.2.0"
TEMPERATURE = 1.0
DEFAULT_SENTENCE_K_MAX = 3
DEFAULT_FIXED_L_VALUES = (1, 10, 20, 40, 60)

DEFAULT_CSV_URL = (
    "https://raw.githubusercontent.com/taksagawa/Paper2026/"
    "main/gpt2/monte_carlo_output/raw_results_generated_samples.csv"
)

REQUIRED_CSV_FIELDS = {
    "sample_index",
    "T",
    "T_blk",
    "n_blocks",
    "sigma_block",
    "sigma_token_Tprime",
    "token_ids",
}

# These values identify the Figure 3 data currently published with the paper.
EXPECTED_RAW_BLOCK_MEAN_PER_TOKEN = 0.4689587089418867
EXPECTED_RAW_TOKEN_TPRIME_MEAN_PER_TOKEN = 4.055417202692808


@dataclass(frozen=True)
class Sample:
    """One valid sequence used in the Figure 3 block analysis."""

    sample_index: int
    full_token_ids: tuple[int, ...]
    sequence: tuple[int, ...]
    T: int
    T_prime: int
    n_sentences: int
    raw_sigma_block: float
    raw_sigma_token_tprime: float


@dataclass(frozen=True)
class AnalysisTask:
    """One reordered sequence whose likelihood must be evaluated."""

    scheme: str
    parameter_name: str
    parameter_value: int
    effective_block_length_tokens: float
    sample_index: int
    transformed_sequence: tuple[int, ...]
    n_source_units: int
    n_reversal_blocks: int
    is_identity: bool


def _strip_ipykernel_arguments(argv: Sequence[str]) -> list[str]:
    """Remove only the ``-f <kernel.json>`` argument injected by Jupyter.

    Jupyter kernels start Python with an argument such as::

        -f /tmp/jupyter-runtime-USER/kernel-UUID.json

    If this file is executed from a notebook cell (rather than with ``%run``),
    that kernel argument remains in ``sys.argv`` and would otherwise make
    ``argparse`` exit with status 2. Unknown user arguments are deliberately
    left untouched so that command-line typos still produce an error.
    """

    filtered: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "-f" and index + 1 < len(argv):
            candidate = argv[index + 1]
            if candidate.endswith(".json") and (
                "kernel-" in candidate or "jupyter-runtime" in candidate
            ):
                index += 2
                continue
        if argument.startswith("-f="):
            candidate = argument[3:]
            if candidate.endswith(".json") and (
                "kernel-" in candidate or "jupyter-runtime" in candidate
            ):
                index += 1
                continue
        filtered.append(argument)
        index += 1
    return filtered


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze GPT-2 entropy production as a function of sentence- and "
            "fixed-token block scale."
        )
    )
    parser.add_argument(
        "--csv-url",
        default=DEFAULT_CSV_URL,
        help="Raw GitHub URL of raw_results_generated_samples.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("block_scale_output"),
        help="Directory for figures, CSV files, metadata, and the downloaded input.",
    )
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument(
        "--model-revision",
        default="main",
        help="Hugging Face model revision. Pin a commit for archival reproducibility.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load GPT-2 only from the local Hugging Face cache.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device, e.g. cuda, cuda:0, cpu, or auto. Default: cuda.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Likelihood-evaluation batch size. Reduce this if CUDA runs out of memory.",
    )
    parser.add_argument(
        "--sentence-k-max",
        type=int,
        default=DEFAULT_SENTENCE_K_MAX,
        help="Largest analyzed sentence-superblock size. Default: 3.",
    )
    parser.add_argument(
        "--fixed-l-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_FIXED_L_VALUES),
        metavar="L",
        help="Analyzed fixed-token block lengths. Default: 1 10 20 40 60.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=2000,
        help="Number of percentile-bootstrap resamples for each plotted mean.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-timeout", type=float, default=30.0)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument(
        "--expected-total-samples",
        type=int,
        default=516,
        help="Expected CSV row count; set to 0 to disable this check.",
    )
    parser.add_argument(
        "--expected-valid-samples",
        type=int,
        default=500,
        help="Expected punctuation-valid sample count; set to 0 to disable this check.",
    )
    parser.add_argument(
        "--reference-tolerance",
        type=float,
        default=0.05,
        help=(
            "Maximum permitted absolute difference in total sigma between the "
            "recomputed k=1/l=1 values and the saved reference values."
        ),
    )
    parser.add_argument(
        "--skip-reference-check",
        action="store_true",
        help="Do not enforce the saved Figure 3 baseline checks.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run fast block-transformation unit tests and exit.",
    )
    arguments = list(sys.argv[1:] if argv is None else argv)
    arguments = _strip_ipykernel_arguments(arguments)
    return parser.parse_args(arguments)


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.sentence_k_max < 1:
        raise ValueError("--sentence-k-max must be at least 1.")
    if not args.fixed_l_values:
        raise ValueError("--fixed-l-values must contain at least one value.")
    if any(value < 1 for value in args.fixed_l_values):
        raise ValueError("Every --fixed-l-values entry must be at least 1.")
    if len(set(args.fixed_l_values)) != len(args.fixed_l_values):
        raise ValueError("--fixed-l-values entries must not be duplicated.")
    if args.bootstrap_resamples < 1:
        raise ValueError("--bootstrap-resamples must be at least 1.")
    if args.download_retries < 1:
        raise ValueError("--download-retries must be at least 1.")
    if args.download_timeout <= 0:
        raise ValueError("--download-timeout must be positive.")
    if args.reference_tolerance < 0:
        raise ValueError("--reference-tolerance must be non-negative.")


def download_bytes(url: str, timeout: float, retries: int) -> bytes:
    """Download a URL with a short retry loop and a fixed user agent."""

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "gpt2-block-scale-analysis/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise RuntimeError(f"HTTP status {status} for {url}")
                data = response.read()
            if not data:
                raise RuntimeError(f"Downloaded an empty file from {url}")
            return data
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt < retries:
                delay = min(2 ** (attempt - 1), 8)
                print(
                    f"Download attempt {attempt}/{retries} failed: {exc}. "
                    f"Retrying in {delay} s...",
                    file=sys.stderr,
                )
                time.sleep(delay)

    raise RuntimeError(f"Failed to download {url} after {retries} attempts") from last_error


def parse_samples(
    csv_bytes: bytes,
    expected_total: int,
    expected_valid: int,
) -> tuple[list[Sample], int]:
    """Parse the downloaded CSV and retain the block-valid samples."""

    try:
        text = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The downloaded CSV is not valid UTF-8.") from exc

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("The downloaded file has no CSV header.")

    missing_fields = REQUIRED_CSV_FIELDS.difference(reader.fieldnames)
    if missing_fields:
        raise ValueError(f"CSV is missing required fields: {sorted(missing_fields)}")

    rows = list(reader)
    if expected_total and len(rows) != expected_total:
        raise ValueError(
            f"Expected {expected_total} CSV rows, but downloaded {len(rows)}. "
            "Check that the URL points to the Figure 3 Monte Carlo data."
        )

    samples: list[Sample] = []
    seen_indices: set[int] = set()
    for row_number, row in enumerate(rows, start=2):
        try:
            sample_index = int(row["sample_index"])
            T = int(row["T"])
            token_list = json.loads(row["token_ids"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid core value in CSV row {row_number}.") from exc

        if sample_index in seen_indices:
            raise ValueError(f"Duplicate sample_index={sample_index} in the CSV.")
        seen_indices.add(sample_index)

        if not isinstance(token_list, list) or not all(
            isinstance(token_id, int) for token_id in token_list
        ):
            raise ValueError(f"token_ids is not a list of integers in CSV row {row_number}.")
        if len(token_list) != T:
            raise ValueError(
                f"CSV row {row_number}: len(token_ids)={len(token_list)} but T={T}."
            )

        # Rows without a valid punctuation-truncated sequence are not part of
        # the Figure 3 block analysis and are deliberately excluded here.
        if not row["T_blk"].strip():
            if row["n_blocks"].strip() or row["sigma_block"].strip():
                raise ValueError(
                    f"CSV row {row_number} has inconsistent empty block fields."
                )
            continue

        try:
            T_prime = int(row["T_blk"])
            n_sentences = int(row["n_blocks"])
            raw_sigma_block = float(row["sigma_block"])
            raw_sigma_token_tprime = float(row["sigma_token_Tprime"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid block value in CSV row {row_number}.") from exc

        if not 1 <= T_prime <= T:
            raise ValueError(f"CSV row {row_number}: invalid T_blk={T_prime} for T={T}.")
        if n_sentences < 1:
            raise ValueError(f"CSV row {row_number}: n_blocks must be positive.")

        full_ids = tuple(token_list)
        samples.append(
            Sample(
                sample_index=sample_index,
                full_token_ids=full_ids,
                sequence=full_ids[:T_prime],
                T=T,
                T_prime=T_prime,
                n_sentences=n_sentences,
                raw_sigma_block=raw_sigma_block,
                raw_sigma_token_tprime=raw_sigma_token_tprime,
            )
        )

    if expected_valid and len(samples) != expected_valid:
        raise ValueError(
            f"Expected {expected_valid} block-valid samples, but found {len(samples)}."
        )
    if not samples:
        raise ValueError("No block-valid samples were found.")

    samples.sort(key=lambda sample: sample.sample_index)
    return samples, len(rows)


def resolve_device(device_argument: str) -> torch.device:
    """Resolve the requested device, requiring CUDA by default."""

    if device_argument == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_argument)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build and verify the NVIDIA driver. "
            "Use --device cpu only for a slow diagnostic run."
        )
    return device


def load_model_and_tokenizer(
    model_name: str,
    revision: str,
    local_files_only: bool,
    device: torch.device,
) -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """Load GPT-2 in float32 for reproducible likelihood comparisons."""

    print(f"Loading tokenizer and model: {model_name} (revision={revision})")
    tokenizer = GPT2Tokenizer.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
    )
    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
    )
    model.to(device=device, dtype=torch.float32)
    model.eval()
    model.config.use_cache = False

    if tokenizer.eos_token_id is None:
        raise RuntimeError("The tokenizer has no EOS/BOS token ID.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Avoid hardware-dependent TF32 rounding in the reference comparison.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

    return model, tokenizer


def build_punctuation_token_ids(
    tokenizer: GPT2Tokenizer,
    samples: Sequence[Sample],
) -> set[int]:
    """Reproduce the punctuation-token rule used by gpt2_sampling.py."""

    observed_ids = {token_id for sample in samples for token_id in sample.sequence}
    punctuation_ids: set[int] = set()
    for token_id in observed_ids:
        decoded = tokenizer.decode([token_id])
        stripped = decoded.rstrip()
        if stripped and stripped[-1] in ".!?":
            punctuation_ids.add(token_id)
    return punctuation_ids


def split_sentence_blocks(
    token_ids: Sequence[int],
    punctuation_ids: set[int],
) -> tuple[tuple[int, ...], ...]:
    """Split a token-ID sequence after each sentence-final punctuation token."""

    blocks: list[tuple[int, ...]] = []
    start = 0
    for index, token_id in enumerate(token_ids):
        if token_id in punctuation_ids:
            blocks.append(tuple(token_ids[start : index + 1]))
            start = index + 1
    if start < len(token_ids):
        blocks.append(tuple(token_ids[start:]))
    return tuple(blocks)


def reverse_sentence_superblocks(
    sentence_blocks: Sequence[Sequence[int]],
    k: int,
) -> tuple[tuple[int, ...], int]:
    """Group k consecutive sentences, then reverse the superblock order."""

    if k < 1:
        raise ValueError("k must be at least 1.")
    superblocks = [sentence_blocks[i : i + k] for i in range(0, len(sentence_blocks), k)]
    transformed = tuple(
        token_id
        for superblock in reversed(superblocks)
        for sentence in superblock
        for token_id in sentence
    )
    return transformed, len(superblocks)


def reverse_fixed_token_blocks(
    token_ids: Sequence[int],
    block_length: int,
) -> tuple[tuple[int, ...], int]:
    """Split from the start into blocks of l tokens and reverse block order."""

    if block_length < 1:
        raise ValueError("block_length must be at least 1.")
    blocks = [token_ids[i : i + block_length] for i in range(0, len(token_ids), block_length)]
    transformed = tuple(token_id for block in reversed(blocks) for token_id in block)
    return transformed, len(blocks)


def assert_token_permutation(
    original: Sequence[int],
    transformed: Sequence[int],
    context: str,
) -> None:
    if len(original) != len(transformed):
        raise AssertionError(f"{context}: the transformation changed sequence length.")
    if Counter(original) != Counter(transformed):
        raise AssertionError(f"{context}: the transformation changed the token multiset.")


def build_fixed_l_values(
    max_length: int,
    requested_values: Sequence[int],
) -> list[int]:
    """Validate and return the explicitly requested fixed-token block lengths."""

    if max_length < 1:
        raise ValueError("max_length must be at least 1.")
    values = list(requested_values)
    if not values:
        raise ValueError("At least one fixed-token block length is required.")
    if any(value < 1 for value in values):
        raise ValueError("Fixed-token block lengths must be at least 1.")
    if len(set(values)) != len(values):
        raise ValueError("Fixed-token block lengths must not be duplicated.")
    if max(values) > max_length:
        raise ValueError(
            "A requested fixed-token block length exceeds the maximum analyzed "
            f"sequence length ({max_length})."
        )
    return sorted(values)


def choose_sentence_k_max(
    requested_k_max: int,
    max_sentences: int,
    mean_sentence_length: float,
    max_sequence_length: int,
) -> int:
    """Apply sentence-count and sequence-length safety limits to requested k."""

    if requested_k_max < 1:
        raise ValueError("requested_k_max must be at least 1.")
    if max_sentences < 1:
        raise ValueError("max_sentences must be at least 1.")
    if not math.isfinite(mean_sentence_length) or mean_sentence_length <= 0:
        raise ValueError("mean_sentence_length must be a positive finite number.")
    if max_sequence_length < 1:
        raise ValueError("max_sequence_length must be at least 1.")

    half_sentence_limit = max(1, max_sentences // 2)
    sequence_length_limit = max(
        1,
        math.floor(max_sequence_length / mean_sentence_length),
    )
    return min(requested_k_max, half_sentence_limit, sequence_length_limit)


def validate_and_segment_samples(
    samples: Sequence[Sample],
    punctuation_ids: set[int],
) -> dict[int, tuple[tuple[int, ...], ...]]:
    """Check CSV/tokenizer agreement and reconstruct every sentence list."""

    sentence_blocks_by_sample: dict[int, tuple[tuple[int, ...], ...]] = {}
    for sample in samples:
        if sample.sequence[-1] not in punctuation_ids:
            raise ValueError(
                f"sample_index={sample.sample_index}: the T' sequence does not end "
                "in a punctuation token under the current tokenizer."
            )
        blocks = split_sentence_blocks(sample.sequence, punctuation_ids)
        if len(blocks) != sample.n_sentences:
            raise ValueError(
                f"sample_index={sample.sample_index}: reconstructed {len(blocks)} "
                f"sentences, but CSV n_blocks={sample.n_sentences}."
            )
        if tuple(token for block in blocks for token in block) != sample.sequence:
            raise AssertionError(
                f"sample_index={sample.sample_index}: sentence blocks do not reconstruct the input."
            )
        sentence_blocks_by_sample[sample.sample_index] = blocks
    return sentence_blocks_by_sample


def build_tasks(
    samples: Sequence[Sample],
    sentence_blocks_by_sample: dict[int, tuple[tuple[int, ...], ...]],
    k_max: int,
    l_values: Sequence[int],
    mean_sentence_length: float,
) -> list[AnalysisTask]:
    """Create all sentence-k and fixed-l transformed sequences."""

    tasks: list[AnalysisTask] = []
    for sample in samples:
        sentence_blocks = sentence_blocks_by_sample[sample.sample_index]

        for k in range(1, k_max + 1):
            transformed, n_superblocks = reverse_sentence_superblocks(sentence_blocks, k)
            assert_token_permutation(
                sample.sequence,
                transformed,
                f"sample={sample.sample_index}, sentence k={k}",
            )
            tasks.append(
                AnalysisTask(
                    scheme="sentence_k",
                    parameter_name="k_sentences",
                    parameter_value=k,
                    effective_block_length_tokens=mean_sentence_length * k,
                    sample_index=sample.sample_index,
                    transformed_sequence=transformed,
                    n_source_units=sample.n_sentences,
                    n_reversal_blocks=n_superblocks,
                    is_identity=(transformed == sample.sequence),
                )
            )

        for block_length in l_values:
            transformed, n_blocks = reverse_fixed_token_blocks(sample.sequence, block_length)
            assert_token_permutation(
                sample.sequence,
                transformed,
                f"sample={sample.sample_index}, fixed l={block_length}",
            )
            tasks.append(
                AnalysisTask(
                    scheme="fixed_token_l",
                    parameter_name="l_tokens",
                    parameter_value=block_length,
                    effective_block_length_tokens=float(block_length),
                    sample_index=sample.sample_index,
                    transformed_sequence=transformed,
                    n_source_units=math.ceil(sample.T_prime / block_length),
                    n_reversal_blocks=n_blocks,
                    is_identity=(transformed == sample.sequence),
                )
            )
    return tasks


def batched_log_likelihoods(
    sequences: Iterable[tuple[int, ...]],
    model: GPT2LMHeadModel,
    device: torch.device,
    bos_token_id: int,
    pad_token_id: int,
    batch_size: int,
) -> dict[tuple[int, ...], float]:
    """Evaluate log P(sequence | BOS) in padded, full-sequence GPU batches."""

    unique_sequences = sorted(set(sequences), key=lambda seq: (len(seq), seq))
    if not unique_sequences:
        return {}
    if any(len(sequence) == 0 for sequence in unique_sequences):
        raise ValueError("Empty sequences are not supported.")

    n_batches = math.ceil(len(unique_sequences) / batch_size)
    results: dict[tuple[int, ...], float] = {}
    print(
        f"Evaluating {len(unique_sequences)} unique sequences in {n_batches} "
        f"batches (batch_size={batch_size}, device={device})..."
    )

    with torch.inference_mode():
        for batch_number, start in enumerate(
            range(0, len(unique_sequences), batch_size),
            start=1,
        ):
            batch_sequences = unique_sequences[start : start + batch_size]
            max_sequence_length = max(len(sequence) for sequence in batch_sequences)
            input_width = max_sequence_length + 1  # BOS + target sequence

            input_ids_cpu = torch.full(
                (len(batch_sequences), input_width),
                fill_value=pad_token_id,
                dtype=torch.long,
            )
            attention_mask_cpu = torch.zeros_like(input_ids_cpu)
            for row_index, sequence in enumerate(batch_sequences):
                sequence_length = len(sequence)
                input_ids_cpu[row_index, 0] = bos_token_id
                input_ids_cpu[row_index, 1 : sequence_length + 1] = torch.tensor(
                    sequence,
                    dtype=torch.long,
                )
                attention_mask_cpu[row_index, : sequence_length + 1] = 1

            input_ids = input_ids_cpu.to(device, non_blocking=(device.type == "cuda"))
            attention_mask = attention_mask_cpu.to(
                device,
                non_blocking=(device.type == "cuda"),
            )

            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                # Position i predicts input token i+1. The final input position
                # has no target and is therefore excluded.
                logits = outputs.logits[:, :-1, :] / TEMPERATURE
                targets = input_ids[:, 1:]
                target_mask = attention_mask[:, 1:].to(dtype=logits.dtype)

                token_nll = F.cross_entropy(
                    logits.transpose(1, 2),
                    targets,
                    reduction="none",
                )
                batch_ll = -(token_nll * target_mask).sum(dim=1)
            except torch.cuda.OutOfMemoryError as exc:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA ran out of memory during likelihood evaluation. "
                    "Rerun with a smaller --batch-size (for example, 8 or 4)."
                ) from exc

            for sequence, value in zip(batch_sequences, batch_ll.cpu().tolist()):
                results[sequence] = float(value)

            del outputs, logits, targets, target_mask, token_nll, batch_ll
            del input_ids, attention_mask

            if batch_number == 1 or batch_number == n_batches or batch_number % 25 == 0:
                print(f"  likelihood batch {batch_number}/{n_batches}")

    return results


def build_per_sample_records(
    samples: Sequence[Sample],
    tasks: Sequence[AnalysisTask],
    likelihoods: dict[tuple[int, ...], float],
) -> list[dict[str, object]]:
    """Combine likelihoods into per-sample entropy-production records."""

    sample_by_index = {sample.sample_index: sample for sample in samples}
    records: list[dict[str, object]] = []
    for task in tasks:
        sample = sample_by_index[task.sample_index]
        forward_ll = likelihoods[sample.sequence]
        reordered_ll = likelihoods[task.transformed_sequence]
        sigma = forward_ll - reordered_ll

        raw_reference_sigma: float | None = None
        if task.scheme == "sentence_k" and task.parameter_value == 1:
            raw_reference_sigma = sample.raw_sigma_block
        elif task.scheme == "fixed_token_l" and task.parameter_value == 1:
            raw_reference_sigma = sample.raw_sigma_token_tprime

        reference_difference = (
            sigma - raw_reference_sigma if raw_reference_sigma is not None else None
        )
        records.append(
            {
                "scheme": task.scheme,
                "parameter_name": task.parameter_name,
                "parameter_value": task.parameter_value,
                "effective_block_length_tokens": task.effective_block_length_tokens,
                "sample_index": sample.sample_index,
                "T_prime": sample.T_prime,
                "n_sentences": sample.n_sentences,
                "n_source_units": task.n_source_units,
                "n_reversal_blocks": task.n_reversal_blocks,
                "is_identity": int(task.is_identity),
                "forward_log_likelihood": forward_ll,
                "reordered_log_likelihood": reordered_ll,
                "sigma": sigma,
                "sigma_per_token": sigma / sample.T_prime,
                "raw_reference_sigma": raw_reference_sigma,
                "reference_difference": reference_difference,
            }
        )
    return records


def reference_validation(
    samples: Sequence[Sample],
    records: Sequence[dict[str, object]],
    tolerance: float,
    enforce: bool,
) -> dict[str, float]:
    """Check k=1 and l=1 against the values saved by gpt2_sampling.py."""

    sample_by_index = {sample.sample_index: sample for sample in samples}
    sentence_differences: list[float] = []
    fixed_differences: list[float] = []
    sentence_values: list[float] = []
    fixed_values: list[float] = []

    for record in records:
        scheme = str(record["scheme"])
        parameter = int(record["parameter_value"])
        if parameter != 1:
            continue
        sample = sample_by_index[int(record["sample_index"])]
        sigma = float(record["sigma"])
        if scheme == "sentence_k":
            sentence_differences.append(sigma - sample.raw_sigma_block)
            sentence_values.append(float(record["sigma_per_token"]))
        elif scheme == "fixed_token_l":
            fixed_differences.append(sigma - sample.raw_sigma_token_tprime)
            fixed_values.append(float(record["sigma_per_token"]))

    if len(sentence_differences) != len(samples) or len(fixed_differences) != len(samples):
        raise AssertionError("The k=1/l=1 reference records are incomplete.")

    raw_sentence_mean = float(
        np.mean([sample.raw_sigma_block / sample.T_prime for sample in samples])
    )
    raw_fixed_mean = float(
        np.mean([sample.raw_sigma_token_tprime / sample.T_prime for sample in samples])
    )

    metrics = {
        "raw_sentence_k1_mean_sigma_per_token": raw_sentence_mean,
        "raw_fixed_l1_mean_sigma_per_token": raw_fixed_mean,
        "recomputed_sentence_k1_mean_sigma_per_token": float(np.mean(sentence_values)),
        "recomputed_fixed_l1_mean_sigma_per_token": float(np.mean(fixed_values)),
        "sentence_k1_mean_abs_sigma_difference": float(
            np.mean(np.abs(sentence_differences))
        ),
        "sentence_k1_max_abs_sigma_difference": float(
            np.max(np.abs(sentence_differences))
        ),
        "fixed_l1_mean_abs_sigma_difference": float(np.mean(np.abs(fixed_differences))),
        "fixed_l1_max_abs_sigma_difference": float(np.max(np.abs(fixed_differences))),
    }

    print("Reference validation:")
    print(
        "  sentence k=1: "
        f"raw mean={raw_sentence_mean:.9f}, "
        f"recomputed mean={metrics['recomputed_sentence_k1_mean_sigma_per_token']:.9f}, "
        f"max |delta sigma|={metrics['sentence_k1_max_abs_sigma_difference']:.6g}"
    )
    print(
        "  fixed l=1:    "
        f"raw mean={raw_fixed_mean:.9f}, "
        f"recomputed mean={metrics['recomputed_fixed_l1_mean_sigma_per_token']:.9f}, "
        f"max |delta sigma|={metrics['fixed_l1_max_abs_sigma_difference']:.6g}"
    )

    if enforce:
        if not math.isclose(
            raw_sentence_mean,
            EXPECTED_RAW_BLOCK_MEAN_PER_TOKEN,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "The downloaded CSV does not reproduce the published sentence-block baseline."
            )
        if not math.isclose(
            raw_fixed_mean,
            EXPECTED_RAW_TOKEN_TPRIME_MEAN_PER_TOKEN,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "The downloaded CSV does not reproduce the published truncated-token baseline."
            )
        max_difference = max(
            metrics["sentence_k1_max_abs_sigma_difference"],
            metrics["fixed_l1_max_abs_sigma_difference"],
        )
        if max_difference > tolerance:
            raise RuntimeError(
                f"Reference recomputation differs from the saved values by as much as "
                f"{max_difference:.6g}, exceeding --reference-tolerance={tolerance}. "
                "Check model/tokenizer revisions, numerical precision, and padding logic."
            )
    return metrics


def bootstrap_mean_ci(
    values: np.ndarray,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    return (
        float(np.percentile(bootstrap_means, 2.5)),
        float(np.percentile(bootstrap_means, 97.5)),
    )


def summarize_records(
    records: Sequence[dict[str, object]],
    bootstrap_resamples: int,
    seed: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        key = (str(record["scheme"]), int(record["parameter_value"]))
        grouped[key].append(record)

    scheme_order = {"sentence_k": 0, "fixed_token_l": 1}
    summaries: list[dict[str, object]] = []
    for scheme, parameter in sorted(
        grouped,
        key=lambda key: (scheme_order[key[0]], key[1]),
    ):
        group = grouped[(scheme, parameter)]
        sigma = np.array([float(record["sigma"]) for record in group], dtype=float)
        sigma_per_token = np.array(
            [float(record["sigma_per_token"]) for record in group],
            dtype=float,
        )
        ci_seed = seed + (100_000 if scheme == "fixed_token_l" else 0) + parameter
        ci_lower, ci_upper = bootstrap_mean_ci(
            sigma_per_token,
            bootstrap_resamples,
            ci_seed,
        )
        n = len(group)
        summaries.append(
            {
                "scheme": scheme,
                "parameter_name": group[0]["parameter_name"],
                "parameter_value": parameter,
                "effective_block_length_tokens": float(
                    group[0]["effective_block_length_tokens"]
                ),
                "N": n,
                "n_identity": int(sum(int(record["is_identity"]) for record in group)),
                "identity_fraction": float(
                    np.mean([int(record["is_identity"]) for record in group])
                ),
                "mean_sigma": float(sigma.mean()),
                "std_sigma": float(sigma.std(ddof=1)),
                "se_sigma": float(sigma.std(ddof=1) / math.sqrt(n)),
                "mean_sigma_per_token": float(sigma_per_token.mean()),
                "std_sigma_per_token": float(sigma_per_token.std(ddof=1)),
                "se_sigma_per_token": float(
                    sigma_per_token.std(ddof=1) / math.sqrt(n)
                ),
                "ci_lower_2.5_sigma_per_token": ci_lower,
                "ci_upper_97.5_sigma_per_token": ci_upper,
                "median_sigma_per_token": float(np.median(sigma_per_token)),
                "min_sigma_per_token": float(sigma_per_token.min()),
                "max_sigma_per_token": float(sigma_per_token.max()),
            }
        )
    return summaries


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(
    summaries: Sequence[dict[str, object]],
    mean_sentence_length: float,
    n_samples: int,
    output_png: Path,
    output_pdf: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    styles = {
        "sentence_k": {
            "label": r"$k$-sentence superblocks",
            "color": "#1F77B4",
            "marker": "o",
            "linestyle": "-",
        },
        "fixed_token_l": {
            "label": r"Fixed token blocks $l$",
            "color": "#D55E00",
            "marker": "s",
            "linestyle": "--",
        },
    }

    for scheme in ("sentence_k", "fixed_token_l"):
        series = [row for row in summaries if row["scheme"] == scheme]
        series.sort(key=lambda row: int(row["parameter_value"]))
        x = np.array(
            [float(row["effective_block_length_tokens"]) for row in series]
        )
        y = np.array([float(row["mean_sigma_per_token"]) for row in series])
        lower = np.array(
            [float(row["ci_lower_2.5_sigma_per_token"]) for row in series]
        )
        upper = np.array(
            [float(row["ci_upper_97.5_sigma_per_token"]) for row in series]
        )
        yerr = np.vstack((y - lower, upper - y))
        style = styles[scheme]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=7.0,
            capsize=6.0,
            capthick=1.8,
            elinewidth=1.8,
            zorder=3,
        )

    ax.axhline(0.0, color="gray", linewidth=1.0)
    ax.set_xlim(left=0.0)
    ax.set_xlabel("Effective block length (tokens)", fontsize=20)
    ax.set_ylabel(r"Mean per-token entropy production $\langle \sigma/T' \rangle$", fontsize=16)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=16, frameon=True)
    ax.text(
        0.98,
        0.96,
        (
            rf"$l'={mean_sentence_length:.2f}$ tokens/sentence" "\n"
            rf"$N={n_samples}$; 95% bootstrap CI"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=16,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )

    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def environment_metadata(device: torch.device) -> dict[str, object]:
    gpu: dict[str, object] | None = None
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        gpu = {
            "index": device_index,
            "name": torch.cuda.get_device_name(device_index),
            "compute_capability": list(torch.cuda.get_device_capability(device_index)),
            "total_memory_bytes": int(properties.total_memory),
        }
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "device": str(device),
        "gpu": gpu,
    }


def run_self_tests() -> None:
    sentence_blocks = ((1, 2), (3,), (4, 5), (6,), (7, 8))
    transformed, n_superblocks = reverse_sentence_superblocks(sentence_blocks, 2)
    assert transformed == (7, 8, 4, 5, 6, 1, 2, 3)
    assert n_superblocks == 3
    assert_token_permutation(
        tuple(token for sentence in sentence_blocks for token in sentence),
        transformed,
        "sentence self-test",
    )

    transformed_fixed, n_fixed_blocks = reverse_fixed_token_blocks((1, 2, 3, 4, 5), 2)
    assert transformed_fixed == (5, 3, 4, 1, 2)
    assert n_fixed_blocks == 3
    assert_token_permutation((1, 2, 3, 4, 5), transformed_fixed, "fixed self-test")

    assert build_fixed_l_values(120, [1, 10, 20, 40, 60]) == [1, 10, 20, 40, 60]
    assert build_fixed_l_values(21, [20, 1, 10]) == [1, 10, 20]
    assert choose_sentence_k_max(3, 19, 22.546782291, 120) == 3
    assert choose_sentence_k_max(7, 19, 22.546782291, 120) == 5
    assert 19 // 2 == 9
    print("Self-tests passed.")


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.self_test:
        run_self_tests()
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Figure 3 samples from:\n  {args.csv_url}")
    csv_bytes = download_bytes(
        args.csv_url,
        timeout=args.download_timeout,
        retries=args.download_retries,
    )
    csv_sha256 = hashlib.sha256(csv_bytes).hexdigest()
    downloaded_csv_path = args.output_dir / "input_raw_results_generated_samples.csv"
    downloaded_csv_path.write_bytes(csv_bytes)
    print(f"Downloaded {len(csv_bytes):,} bytes; SHA-256={csv_sha256}")

    samples, total_sample_count = parse_samples(
        csv_bytes,
        expected_total=args.expected_total_samples,
        expected_valid=args.expected_valid_samples,
    )
    print(
        f"Parsed {total_sample_count} total samples; "
        f"using {len(samples)} punctuation-valid T' sequences."
    )

    device = resolve_device(args.device)
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        args.model_revision,
        args.local_files_only,
        device,
    )

    punctuation_ids = build_punctuation_token_ids(tokenizer, samples)
    sentence_blocks_by_sample = validate_and_segment_samples(samples, punctuation_ids)

    total_truncated_tokens = sum(sample.T_prime for sample in samples)
    total_sentences = sum(sample.n_sentences for sample in samples)
    mean_sentence_length = total_truncated_tokens / total_sentences
    max_sentences = max(sample.n_sentences for sample in samples)
    max_length = max(sample.T_prime for sample in samples)
    half_sentence_k_limit = max(1, max_sentences // 2)
    sequence_length_k_limit = max(
        1,
        math.floor(max_length / mean_sentence_length),
    )
    # Apply the explicit plotting range while retaining the two safety limits.
    k_max = choose_sentence_k_max(
        args.sentence_k_max,
        max_sentences,
        mean_sentence_length,
        max_length,
    )
    l_values = build_fixed_l_values(max_length, args.fixed_l_values)

    print(f"Pooled mean sentence length l' = {mean_sentence_length:.9f} tokens")
    print(
        f"Observed max sentence count = {max_sentences}; "
        f"half-count k limit = {half_sentence_k_limit}."
    )
    print(
        f"Observed max truncated length T' = {max_length}; "
        f"sequence-length k limit = {sequence_length_k_limit}."
    )
    print(
        f"Requested sentence range = k=1..{args.sentence_k_max}; "
        f"using k=1..{k_max} (rightmost k*l'={k_max * mean_sentence_length:.6f} "
        f"<= max T'={max_length})."
    )
    print(f"Fixed token block lengths: {l_values}")

    tasks = build_tasks(
        samples,
        sentence_blocks_by_sample,
        k_max,
        l_values,
        mean_sentence_length,
    )
    all_sequences = [sample.sequence for sample in samples]
    all_sequences.extend(task.transformed_sequence for task in tasks)

    likelihoods = batched_log_likelihoods(
        all_sequences,
        model,
        device,
        bos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=args.batch_size,
    )
    per_sample_records = build_per_sample_records(samples, tasks, likelihoods)

    validation_metrics = reference_validation(
        samples,
        per_sample_records,
        tolerance=args.reference_tolerance,
        enforce=not args.skip_reference_check,
    )
    summaries = summarize_records(
        per_sample_records,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )

    per_sample_records.sort(
        key=lambda row: (
            0 if row["scheme"] == "sentence_k" else 1,
            int(row["parameter_value"]),
            int(row["sample_index"]),
        )
    )
    summary_csv_path = args.output_dir / "block_scale_summary.csv"
    per_sample_csv_path = args.output_dir / "block_scale_per_sample.csv"
    figure_png_path = args.output_dir / "block_scale_comparison.png"
    figure_pdf_path = args.output_dir / "block_scale_comparison.pdf"
    metadata_path = args.output_dir / "analysis_metadata.json"

    write_csv(per_sample_csv_path, per_sample_records)
    write_csv(summary_csv_path, summaries)
    plot_summary(
        summaries,
        mean_sentence_length,
        len(samples),
        figure_png_path,
        figure_pdf_path,
    )

    # ``__file__`` is not defined when the script is pasted directly into a
    # Jupyter cell. Use the stable script filename as a notebook-safe fallback.
    script_filename = Path(
        globals().get("__file__") or "gpt2_block_scale_analysis.py"
    ).name

    metadata = {
        "script": {
            "name": script_filename,
            "version": SCRIPT_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "input": {
            "csv_url": args.csv_url,
            "downloaded_csv": str(downloaded_csv_path.resolve()),
            "csv_size_bytes": len(csv_bytes),
            "csv_sha256": csv_sha256,
            "n_total_samples": total_sample_count,
            "n_valid_samples": len(samples),
        },
        "model": {
            "name": args.model_name,
            "requested_revision": args.model_revision,
            "resolved_commit_hash": getattr(model.config, "_commit_hash", None),
            "temperature": TEMPERATURE,
            "dtype": "float32",
            "bos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "analysis": {
            "sequence_basis": "same punctuation-valid T_prime sequences for both schemes",
            "normalization": "per-sample sigma/T_prime, then average across samples",
            "mean_sentence_length_definition": "sum(T_prime) / sum(n_sentences)",
            "mean_sentence_length_tokens": mean_sentence_length,
            "total_truncated_tokens": total_truncated_tokens,
            "total_sentences": total_sentences,
            "max_observed_sentences": max_sentences,
            "max_truncated_length_tokens": max_length,
            "requested_sentence_k_max": args.sentence_k_max,
            "sentence_x_rightmost_tokens": k_max * mean_sentence_length,
            "half_sentence_k_limit": half_sentence_k_limit,
            "sequence_length_k_limit": sequence_length_k_limit,
            "k_max_rule": (
                "min(requested_sentence_k_max, "
                "max(1, floor(max_observed_sentences / 2)), "
                "max(1, floor(max_truncated_length_tokens / mean_sentence_length_tokens)))"
            ),
            "k_max": k_max,
            "requested_fixed_l_values": list(args.fixed_l_values),
            "fixed_l_values": l_values,
            "fixed_l1_handling": "included in figures and CSV outputs",
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.seed,
            "batch_size": args.batch_size,
        },
        "reference_validation": validation_metrics,
        "environment": environment_metadata(device),
        "outputs": {
            "summary_csv": str(summary_csv_path.resolve()),
            "per_sample_csv": str(per_sample_csv_path.resolve()),
            "figure_png": str(figure_png_path.resolve()),
            "figure_pdf": str(figure_pdf_path.resolve()),
        },
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print("\nAnalysis completed.")
    print(f"  summary:    {summary_csv_path.resolve()}")
    print(f"  per-sample: {per_sample_csv_path.resolve()}")
    print(f"  figure:     {figure_png_path.resolve()}")
    print(f"  metadata:   {metadata_path.resolve()}")
    return 0


if __name__ == "__main__":
    _exit_code = main()
    # Raising SystemExit is appropriate for a terminal command, but IPython
    # displays even SystemExit(0) as a traceback-like exception in a notebook.
    if "ipykernel" not in sys.modules:
        raise SystemExit(_exit_code)
