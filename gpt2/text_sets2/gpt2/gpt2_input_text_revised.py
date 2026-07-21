# =============================================================================
# Entropy Production for Autoregressive Models: GPT-2 Numerical Experiment
# Local / standard Python version (GPU optional; CPU is also supported)
# =============================================================================
# Paper: "Stochastic Thermodynamics for Autoregressive Generative Models:
# A Non-Markovian Perspective"
# Estimation of entropy production — proof-of-concept demonstration
#
# The model argument can be either a Hugging Face model ID or a local directory
# =============================================================================

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import warnings
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


DEFAULT_TEXTS_URL = (
    "https://raw.githubusercontent.com/taksagawa/Paper2026/main/"
    "gpt2/text_sets2/fixed_text_GPT56.json"
)


def _running_as_kernel_launcher():
    """Return True when code is pasted/executed in a live Jupyter kernel.

    Under ``%run script.py ...``, ``sys.argv[0]`` is the script path, so
    explicit script arguments remain available.  When code is executed
    directly in a notebook cell, ``sys.argv[0]`` is normally
    ``ipykernel_launcher.py`` and includes Jupyter's internal ``-f`` option.
    """
    launcher = Path(sys.argv[0]).name.lower() if sys.argv else ""
    return launcher.startswith("ipykernel_launcher") or launcher.startswith("jupyter-kernel")


def _remove_ipykernel_arguments(argv):
    """Remove Jupyter's internal connection-file arguments.

    This is intentionally independent of imported-module detection because
    some Jupyter installations expose ``ipykernel_launcher.py`` in sys.argv
    before the ``ipykernel`` package appears in ``sys.modules``.
    """
    cleaned = []
    i = 0
    while i < len(argv):
        arg = str(argv[i])
        if arg == "-f" and i + 1 < len(argv):
            i += 2
            continue
        if arg.startswith("-f=") or arg.startswith("--f="):
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    return cleaned


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute token- and block-reversal entropy production with GPT-2."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GPT2_MODEL", "gpt2"),
        help="Hugging Face model ID or local model directory (default: gpt2).",
    )
    parser.add_argument(
        "--texts",
        type=Path,
        default=Path(os.environ.get("TEXTS_JSON", "fixed_text_GPT56.json")),
        help="Local JSON file containing causal_texts and noncausal_texts.",
    )
    parser.add_argument(
        "--texts-url",
        default=DEFAULT_TEXTS_URL,
        help="Fallback URL used only when --texts does not exist.",
    )
    parser.add_argument(
        "--no-download-texts",
        action="store_true",
        help="Fail instead of downloading the JSON when --texts does not exist.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", "fixed_text_output")),
        help="Directory for figures, CSV files, metadata, and logs.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default=os.environ.get("DEVICE", "auto"),
        help="Computation device (default: auto).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the model/tokenizer only from the local Hugging Face cache or path.",
    )
    parser.add_argument(
        "--expected-n",
        type=int,
        default=100,
        help="Expected number of texts in each category (default: 100).",
    )
    parser.add_argument(
        "--expected-blocks",
        type=int,
        default=4,
        help="Expected sentence-block count per text; use 0 to disable (default: 4).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    if argv is None:
        argv = sys.argv[1:]
        if _running_as_kernel_launcher():
            argv = _remove_ipykernel_arguments(argv)
    else:
        argv = _remove_ipykernel_arguments(argv)
    return parser.parse_args(argv)


ARGS = parse_args()
OUTPUT_DIR = ARGS.output_dir.expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if any(OUTPUT_DIR.iterdir()):
    print(
        f"Warning: output directory is not empty; files with matching names "
        f"will be overwritten: {OUTPUT_DIR}"
    )

# A noninteractive backend is safer on remote/headless GPU machines.
# Use --show on a desktop if interactive windows are desired.
if not ARGS.show:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Japanese font support (optional)
try:
    import japanize_matplotlib
except ImportError:
    pass

# Device configuration
if ARGS.device == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif ARGS.device == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but CUDA is not available to PyTorch. "
            "Check the NVIDIA driver and the CUDA-enabled PyTorch installation."
        )
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")

# Set and record global random seeds.  The fixed-text likelihood evaluation
# itself performs no sampling; strip-plot jitter uses a separate local RNG below.
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f"Random seed: {RANDOM_SEED}")

# Load GPT-2 from a Hub model ID, local cache, or local directory.
# For private/gated repositories, set HF_TOKEN in the shell environment.
print(f"Loading model/tokenizer: {ARGS.model}")
_load_kwargs = {"local_files_only": ARGS.local_files_only}
tokenizer = GPT2Tokenizer.from_pretrained(
    ARGS.model,
    add_prefix_space=True,
    **_load_kwargs,
)
model = GPT2LMHeadModel.from_pretrained(
    ARGS.model,
    **_load_kwargs,
).to(device)
model.eval()
print("GPT-2 loaded successfully.")

# Cache of punctuation token IDs (computed once)
# Used by split_token_ids_into_blocks
print("Building punctuation token cache...")
PUNCT_TOKEN_IDS = set()
for _tid in range(tokenizer.vocab_size):
    _decoded = tokenizer.decode([_tid])
    if _decoded.rstrip() and _decoded.rstrip()[-1] in '.!?':
        PUNCT_TOKEN_IDS.add(_tid)
print(f"  {len(PUNCT_TOKEN_IDS)} punctuation tokens found.")

# %% ── Cell 2: Core functions ──

def compute_log_likelihood(token_ids, model, device, tokenizer):
    """
    Compute the full log-likelihood of a nonempty token sequence, including
    the first-token term conditioned on GPT-2's BOS/EOS token.

    If token_ids = [y_1, ..., y_T], this computes

        L = sum_{t=1}^T log p(y_t | BOS, y_{1:t-1}).

    The target log-probabilities are gathered in one tensor operation on the
    selected device, avoiding a device synchronization for every token.
    """
    token_ids = list(token_ids)
    if not token_ids:
        raise ValueError("token_ids must not be empty.")

    max_positions = getattr(
        model.config,
        "max_position_embeddings",
        getattr(model.config, "n_positions", None),
    )
    if max_positions is None:
        raise AttributeError(
            "The model config has neither max_position_embeddings nor n_positions."
        )
    if len(token_ids) > max_positions:
        raise ValueError(
            f"Token count ({len(token_ids)}) exceeds GPT-2 context length "
            f"({max_positions}). Results would be invalid."
        )

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    if bos_id is None:
        raise ValueError("The tokenizer has neither bos_token_id nor eos_token_id.")

    # T target tokens require T input positions: [BOS, y_1, ..., y_{T-1}].
    input_seq = [bos_id] + token_ids[:-1]
    input_ids = torch.tensor([input_seq], dtype=torch.long, device=device)
    targets = torch.tensor(token_ids, dtype=torch.long, device=device)

    with torch.inference_mode():
        logits = model(input_ids, use_cache=False).logits[0]
        per_token_ll_tensor = (
            torch.nn.functional.log_softmax(logits, dim=-1)
            .gather(dim=-1, index=targets.unsqueeze(-1))
            .squeeze(-1)
        )

    # Accumulate the model-produced per-token log-probabilities in float64 to
    # avoid float32 sequence-sum reduction-order effects.
    per_token_ll = per_token_ll_tensor.cpu().numpy().astype(np.float64, copy=False)
    total_ll = float(per_token_ll.sum(dtype=np.float64))
    return total_ll, per_token_ll


def compute_full_log_likelihood(token_ids, model, device, tokenizer):
    """
    Compute the full path log-likelihood including the initial-distribution term.

    Delegates to compute_log_likelihood, which prepends BOS internally.

    If token_ids = [y_1, ..., y_T], this function computes
        L_full = Σ_{t=1}^{T} ln p(y_t | BOS, y_{1:t-1})

    Returns:
        ll_full:       total log-likelihood (scalar)
        ll_first:      ln p(y_1 | BOS)
        ll_cond:       Σ_{t=2}^{T} ln p(y_t | BOS, y_{1:t-1})
        per_token_ll:  per-token log-likelihood (array of length T)
    """
    total_ll, per_token_ll = compute_log_likelihood(
        token_ids, model, device, tokenizer
    )
    ll_first = per_token_ll[0]
    ll_cond = total_ll - ll_first
    return total_ll, ll_first, ll_cond, per_token_ll


def compute_sigma_token(token_ids, model, device, tokenizer,
                        _precomputed_fwd=None):
    """
    Compute the entropy production via token-level reversal.

    Corresponds to eq. (cg-sigma) with block length l=1:
      σ_token = ln P(y_{1:T}) - ln P(y_T, y_{T-1}, ..., y_1)
    where P is the full path probability including the initial distribution term.

    Both forward and reversed sequences are evaluated with BOS as the first
    model input.  To obtain T target log-probabilities, the actual tensors are:
      forward input:   [BOS, y_1, ..., y_{T-1}]
      forward targets: [y_1, ..., y_T]
      reversed input:  [BOS, y_T, ..., y_2]
      reversed targets:[y_T, ..., y_1]
    This is equivalent, for the required likelihood terms, to passing the full
    BOS-prefixed sequence and discarding the final unused next-token logit.

    The boundary correction (difference between initial-token log-probs)
    is implicitly included in sigma and also returned separately for
    diagnostic purposes.

    If _precomputed_fwd is given as a 4-tuple
    (ll_full, ll_first, ll_cond, per_token_ll), the forward pass is
    skipped and these values are reused (avoids redundant computation
    when the caller has already evaluated the forward log-likelihood).
    """
    # Forward (full log-likelihood with BOS)
    if _precomputed_fwd is not None:
        ll_fwd_full, ll_first_fwd, ll_fwd_cond, per_token_fwd = _precomputed_fwd
    else:
        ll_fwd_full, ll_first_fwd, ll_fwd_cond, per_token_fwd = \
            compute_full_log_likelihood(token_ids, model, device, tokenizer)

    # Token-reversed (full log-likelihood with BOS)
    reversed_ids = list(reversed(token_ids))
    ll_rev_full, ll_first_rev, ll_rev_cond, per_token_rev = \
        compute_full_log_likelihood(reversed_ids, model, device, tokenizer)

    sigma = ll_fwd_full - ll_rev_full
    boundary_correction = ll_first_fwd - ll_first_rev  # diagnostic only

    return sigma, ll_fwd_full, ll_rev_full, per_token_fwd, per_token_rev, boundary_correction


def split_token_ids_into_blocks(token_ids, tokenizer):
    """
    Split a token-ID sequence into sentence blocks.
    To satisfy eq. (block-rev-def), segmentation is performed at the
    token-ID level, not at the text-string level.

    Strategy: split at tokens whose decoded text ends with sentence-final punctuation (. ! ?).
    This avoids BPE re-tokenization issues.  When the final token is a delimiter
    (checked by compute_sigma_block), deterministic block-order reversal is an
    involution and hence a bijection on the supported token sequences.
    Uses the PUNCT_TOKEN_IDS cache built in Cell 1.
    """
    # Find block boundaries
    blocks = []
    current_block_start = 0

    for i, tid in enumerate(token_ids):
        if tid in PUNCT_TOKEN_IDS:
            # End block including the punctuation token
            blocks.append(token_ids[current_block_start:i+1])
            current_block_start = i + 1

    # Remaining tokens without trailing punctuation
    if current_block_start < len(token_ids):
        blocks.append(token_ids[current_block_start:])

    return blocks


def compute_sigma_block(text, token_ids, model, device, tokenizer,
                        _precomputed_fwd=None):
    """
    Compute the entropy production via block (sentence) reversal.

    Paper eq. (cg-sigma):
      σ_block = ln P(y_{1:T}) - ln P(ỹ'_{1:T'})
    where tilde{y}' is the token sequence with reversed block order.

    Important: block reversal follows eq. (block-rev-def) and is performed
    at the token-ID level; no text-level re-tokenization is performed.
    Together with the delimiter condition below, this makes block-order reversal
    deterministic and invertible on the supported token sequences.

    Requires that the last token is a sentence-final punctuation token
    (delimiter condition for the bijection; see the paper).

    If _precomputed_fwd is given as a 4-tuple
    (ll_full, ll_first, ll_cond, per_token_ll), the forward pass is
    skipped and these values are reused.

    Returns:
        sigma, ll_fwd_full, ll_rev_full, text_blocks, reversed_text_blocks,
        n_token_blocks  (number of token-level blocks)
    """
    # Verify delimiter condition: y_T must be a punctuation token
    if token_ids[-1] not in PUNCT_TOKEN_IDS:
        raise ValueError(
            f"Bijection condition violated: last token {token_ids[-1]} "
            f"('{tokenizer.decode([token_ids[-1]])}') is not a delimiter. "
            f"The sequence must end with sentence-final punctuation."
        )
    # Forward full log-likelihood
    if _precomputed_fwd is not None:
        ll_fwd_full, ll_first_fwd, ll_fwd_cond, _ = _precomputed_fwd
    else:
        ll_fwd_full, ll_first_fwd, ll_fwd_cond, _ = \
            compute_full_log_likelihood(token_ids, model, device, tokenizer)

    # Block segmentation at the token level
    token_blocks = split_token_ids_into_blocks(token_ids, tokenizer)
    # Derive display text blocks from token blocks (guarantees 1-to-1 correspondence)
    text_blocks = [tokenizer.decode(blk) for blk in token_blocks]

    if len(token_blocks) <= 1:
        # Only one block; reversal is identity -> sigma=0
        return 0.0, ll_fwd_full, ll_fwd_full, text_blocks, text_blocks, len(token_blocks)

    # Reverse block order at the token level (preserve intra-block token order)
    # Paper eq. (block-rev-def): tilde{y}' = (B_k, B_{k-1}, ..., B_1)
    reversed_token_blocks = list(reversed(token_blocks))
    reversed_token_ids = []
    for block in reversed_token_blocks:
        reversed_token_ids.extend(block)

    # Full log-likelihood of the reversed sequence
    ll_rev_full, _, _, _ = \
        compute_full_log_likelihood(reversed_token_ids, model, device, tokenizer)

    sigma = ll_fwd_full - ll_rev_full

    # Reversed text blocks for display
    reversed_text_blocks = [tokenizer.decode(blk) for blk in reversed_token_blocks]

    return sigma, ll_fwd_full, ll_rev_full, text_blocks, reversed_text_blocks, len(token_blocks)


# %% ── Cell 3: Text data preparation ──

TEXTS_URL = ARGS.texts_url
texts_path = ARGS.texts.expanduser().resolve()
texts_data = None
texts_source = None

if texts_path.exists():
    raw_texts = texts_path.read_bytes()
    texts_data = json.loads(raw_texts.decode("utf-8"))
    texts_source = str(texts_path)
    print(f"Loaded text data from local file: {texts_path}")
elif ARGS.no_download_texts:
    raise FileNotFoundError(
        f"Text JSON was not found: {texts_path}\n"
        "Provide it with --texts, or omit --no-download-texts to use the URL fallback."
    )
else:
    print(f"Local text file not found; downloading from: {TEXTS_URL}")
    try:
        with urllib.request.urlopen(TEXTS_URL, timeout=120) as response:
            raw_texts = response.read()
        texts_data = json.loads(raw_texts.decode("utf-8"))
        texts_path.parent.mkdir(parents=True, exist_ok=True)
        texts_path.write_bytes(raw_texts)
        texts_source = TEXTS_URL
        print(f"Downloaded text data and cached it at: {texts_path}")
    except Exception as exc:
        raise RuntimeError(
            f"Could not load text data from either {texts_path} or {TEXTS_URL}."
        ) from exc

if not isinstance(texts_data, dict):
    raise TypeError("The text JSON root must be an object.")
for required_key in ("causal_texts", "noncausal_texts"):
    if required_key not in texts_data:
        raise KeyError(f"Missing required JSON key: {required_key}")
    if not isinstance(texts_data[required_key], list):
        raise TypeError(f"{required_key} must be a JSON array.")

_text_metadata = texts_data.get("metadata")
if not isinstance(_text_metadata, dict):
    raise TypeError("The text JSON metadata must be an object.")

_texts_filename_stem = texts_path.stem
_texts_filename_prefix, _separator, _TEXT_SET_MODEL_TAG = \
    _texts_filename_stem.rpartition("_")
if not _separator or not _TEXT_SET_MODEL_TAG:
    raise ValueError(
        f"The text JSON filename must end with an underscore suffix, "
        f"such as '_GPT56.json': {texts_path.name!r}"
    )


def _model_tagged_filename(filename):
    """Insert the text-generation model tag before a filename's extension."""
    path = Path(filename)
    return f"{path.stem}_{_TEXT_SET_MODEL_TAG}{path.suffix}"

causal_texts = texts_data["causal_texts"]
noncausal_texts = texts_data["noncausal_texts"]

for category, texts in (("causal", causal_texts), ("non-causal", noncausal_texts)):
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"{category} text {index} must be a nonempty string."
            )

_texts_file_sha256 = hashlib.sha256(raw_texts).hexdigest()
_texts_canonical_bytes = json.dumps(
    texts_data,
    sort_keys=True,
    ensure_ascii=False,
    separators=(",", ":"),
).encode("utf-8")
_texts_canonical_sha256 = hashlib.sha256(_texts_canonical_bytes).hexdigest()

if len(causal_texts) != ARGS.expected_n:
    raise ValueError(
        f"Expected {ARGS.expected_n} causal texts, got {len(causal_texts)}"
    )
if len(noncausal_texts) != ARGS.expected_n:
    raise ValueError(
        f"Expected {ARGS.expected_n} non-causal texts, got {len(noncausal_texts)}"
    )

print(f"Causal texts: {len(causal_texts)} samples")
print(f"Non-causal texts: {len(noncausal_texts)} samples")
print(f"Text-generation model tag: {_TEXT_SET_MODEL_TAG}")

# %% ── Cell 4: Main computation ──

def analyze_texts(texts, label, model, device, tokenizer):
    """Compute sigma_token and sigma_block for a set of texts."""
    results = []

    for i, text in enumerate(texts):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        T = len(token_ids)
        if T == 0:
            raise ValueError(f"{label} text {i} produced an empty token sequence.")

        token_blocks = split_token_ids_into_blocks(token_ids, tokenizer)
        if ARGS.expected_blocks > 0 and len(token_blocks) != ARGS.expected_blocks:
            raise ValueError(
                f"{label} text {i} has {len(token_blocks)} sentence blocks; "
                f"expected {ARGS.expected_blocks}: {text!r}"
            )

        # Forward log-likelihood (computed once, shared by both functions)
        fwd_result = compute_full_log_likelihood(
            token_ids, model, device, tokenizer
        )

        # Token reversal
        sigma_tok, ll_fwd, ll_rev, _, _, _ = compute_sigma_token(
            token_ids, model, device, tokenizer,
            _precomputed_fwd=fwd_result
        )

        # Block reversal
        sigma_blk, _, ll_blk_rev, blocks, rev_blocks, n_blocks = \
            compute_sigma_block(
                text, token_ids, model, device, tokenizer,
                _precomputed_fwd=fwd_result
            )

        results.append({
            "index": i,
            "text": text,
            "T": T,
            "n_blocks": n_blocks,
            "ll_fwd": ll_fwd,
            "ll_token_rev": ll_rev,
            "ll_block_rev": ll_blk_rev,
            "sigma_token": sigma_tok,
            "sigma_block": sigma_blk,
            "sigma_token_per_T": sigma_tok / max(T, 1),
            "sigma_block_per_T": sigma_blk / max(T, 1),
        })

        if i < 3:
            print(f"\n--- [{label}] Sample {i} (T={T}, blocks={n_blocks}) ---")
            print(f"  Text: {text[:80]}...")
            print(f"  L_fwd      = {ll_fwd:.2f}")
            print(f"  L_tok_rev  = {ll_rev:.2f}")
            print(f"  L_blk_rev  = {ll_blk_rev:.2f}")
            print(f"  σ_token    = {sigma_tok:.2f}")
            print(f"  σ_block    = {sigma_blk:.2f}")

    return results


print("=" * 60)
print("Analyzing causal texts")
print("=" * 60)
results_causal = analyze_texts(causal_texts, "Causal", model, device, tokenizer)

print("\n" + "=" * 60)
print("Analyzing non-causal texts")
print("=" * 60)
results_noncausal = analyze_texts(noncausal_texts, "Non-causal", model, device, tokenizer)

# %% ── Cell 5: Sanity checks ──

print("=" * 60)
print("Sanity checks")
print("=" * 60)

# Check 1: single block -> sigma_block = 0
print("\n[Check 1] Single-sentence text → σ_block should be 0")
single_text = "The cat sat on the mat."
single_ids = tokenizer.encode(single_text, add_special_tokens=False)
sigma_blk_single, _, _, _, _, _ = compute_sigma_block(
    single_text, single_ids, model, device, tokenizer
)
print(f"  σ_block = {sigma_blk_single:.6f}  (expected: 0.0)")

# Check 2: sample-level length-preservation and round-trip diagnostics
# The paper requires block reversal R_S to be a bijection on the supported
# sequence space.  Length equality alone is not a proof of bijectivity; the
# following checks verify length preservation and involution for one example.
print("\n[Check 2] Bijection check: |reversed_ids| == |original_ids|")
test_text2 = "She opened her eyes. She got out of bed. She made coffee."
test_ids2 = tokenizer.encode(test_text2, add_special_tokens=False)
token_blocks = split_token_ids_into_blocks(test_ids2, tokenizer)
reversed_token_ids_2 = []
for block in reversed(token_blocks):
    reversed_token_ids_2.extend(block)
bijection_ok = len(reversed_token_ids_2) == len(test_ids2)
# For this example, also verify that double reversal recovers the original.
blocks_of_reversed = split_token_ids_into_blocks(reversed_token_ids_2, tokenizer)
double_reversed = []
for block in reversed(blocks_of_reversed):
    double_reversed.extend(block)
round_trip_ok = double_reversed == test_ids2
print(f"  |original| = {len(test_ids2)}, |reversed| = {len(reversed_token_ids_2)} → {'OK ✓' if bijection_ok else 'FAIL ✗'}")
print(f"  Round-trip (reverse twice = identity): {'OK ✓' if round_trip_ok else 'FAIL ✗'}")

# Check 3: descriptive sample means
# The fixed corpus is not sampled from GPT-2 itself, so no non-negativity
# theorem applies to these empirical means.
sigmas_tok_c = [r["sigma_token"] for r in results_causal]
sigmas_tok_nc = [r["sigma_token"] for r in results_noncausal]
sigmas_blk_c = [r["sigma_block"] for r in results_causal]
sigmas_blk_nc = [r["sigma_block"] for r in results_noncausal]

print("\n[Check 3] Descriptive sample means of sigma")
print("  No non-negativity constraint applies to this fixed corpus.")
print(f"  mean sigma_token (causal)     = {np.mean(sigmas_tok_c):.2f}")
print(f"  mean sigma_token (non-causal) = {np.mean(sigmas_tok_nc):.2f}")
print(f"  mean sigma_block (causal)     = {np.mean(sigmas_blk_c):.2f}")
print(f"  mean sigma_block (non-causal) = {np.mean(sigmas_blk_nc):.2f}")

# %% ── Cell 6: Statistical summary ──

def print_stats(name, values):
    arr = np.array(values)
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    print(f"  {name:40s}: mean={arr.mean():8.2f},  std={arr.std(ddof=1):7.2f},  SE={se:6.2f},  min={arr.min():8.2f},  max={arr.max():8.2f}")

print("\n" + "=" * 60)
print("Statistical summary")
print("=" * 60)

print("\n* Token reversal sigma_token:")
print_stats("Causal", sigmas_tok_c)
print_stats("Non-causal", sigmas_tok_nc)

print("\n* Block reversal sigma_block:")
print_stats("Causal", sigmas_blk_c)
print_stats("Non-causal", sigmas_blk_nc)

print("\n* Per-token sigma/T:")
print_stats("σ_token/T (Causal)", [r["sigma_token_per_T"] for r in results_causal])
print_stats("σ_token/T (Non-causal)", [r["sigma_token_per_T"] for r in results_noncausal])
print_stats("σ_block/T (Causal)", [r["sigma_block_per_T"] for r in results_causal])
print_stats("σ_block/T (Non-causal)", [r["sigma_block_per_T"] for r in results_noncausal])

# %% ── Cell 7: Main visualization (token reversal / block reversal side by side) ──
#
# Display only per-token entropy production (sigma/T).
# Left: token reversal sigma_token/T, Right: block reversal sigma_block/T

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
color_causal = "#E74C3C"
color_noncausal = "#3498DB"

def draw_boxpair(ax, data_causal, data_noncausal, ylabel, title):
    """Draw a box plot comparing causal vs non-causal in one panel."""
    data = [data_causal, data_noncausal]
    labels = ["Causal", "Non-causal"]
    colors = [color_causal, color_noncausal]

    boxplot_kwargs = dict(
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
    )
    try:
        # Matplotlib >= 3.9
        bp = ax.boxplot(data, tick_labels=labels, **boxplot_kwargs)
    except TypeError:
        # Matplotlib <= 3.8
        bp = ax.boxplot(data, labels=labels, **boxplot_kwargs)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    means = [np.mean(d) for d in data]
    ax.scatter([1, 2], means, marker="D", color="black", s=70, zorder=5, label="Mean")

    # Overlay individual data points as a strip plot
    rng = np.random.default_rng(42)
    for j, (d, c) in enumerate(zip(data, colors)):
        jitter = rng.uniform(-0.12, 0.12, len(d))
        ax.scatter(np.full(len(d), j + 1) + jitter, d,
                   c=c, alpha=0.35, s=20, edgecolors="none", zorder=3)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.legend(fontsize=18)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis='x', labelsize=20)

# (a) Token reversal sigma/T
draw_boxpair(axes[0],
             [r["sigma_token_per_T"] for r in results_causal],
             [r["sigma_token_per_T"] for r in results_noncausal],
             ylabel=r"$\sigma_{\mathrm{token}} / T$",
             title="(a) Token reversal")

# (b) Block reversal sigma/T
draw_boxpair(axes[1],
             [r["sigma_block_per_T"] for r in results_causal],
             [r["sigma_block_per_T"] for r in results_noncausal],
             ylabel=r"$\sigma_{\mathrm{block}} / T$",
             title="(b) Block reversal")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / _model_tagged_filename("entropy_production_comparison.png"), dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / _model_tagged_filename("entropy_production_comparison.pdf"), bbox_inches="tight")
if ARGS.show:
    plt.show()
plt.close(fig)
print(f"Figure saved in: {OUTPUT_DIR}")

# %% ── Cell 8: Scatter plot of sigma_token vs sigma_block ──

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Compute shared axis range for both panels (equal x/y range so diagonal is 45 degrees)
all_sigma_tok = [r["sigma_token"] for r in results_causal] + [r["sigma_token"] for r in results_noncausal]
all_sigma_blk = [r["sigma_block"] for r in results_causal] + [r["sigma_block"] for r in results_noncausal]
all_vals = all_sigma_tok + all_sigma_blk
v_min, v_max = min(all_vals), max(all_vals)
v_margin = (v_max - v_min) * 0.05
shared_lim = (v_min - v_margin, v_max + v_margin)

# Causal
ax = axes[0]
ax.scatter(
    [r["sigma_token"] for r in results_causal],
    [r["sigma_block"] for r in results_causal],
    c="#E74C3C", alpha=0.7, s=60, edgecolors="black", linewidths=0.5,
)
ax.plot([shared_lim[0], shared_lim[1]], [shared_lim[0], shared_lim[1]], "k--", alpha=0.3, label="σ_block = σ_token")
ax.set_xlim(shared_lim)
ax.set_ylim(shared_lim)
ax.set_aspect("equal")
ax.set_xlabel(r"$\sigma_{\mathrm{token}}$", fontsize=13)
ax.set_ylabel(r"$\sigma_{\mathrm{block}}$", fontsize=13)
ax.set_title("(a) Causal texts", fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# Non-causal
ax = axes[1]
ax.scatter(
    [r["sigma_token"] for r in results_noncausal],
    [r["sigma_block"] for r in results_noncausal],
    c="#3498DB", alpha=0.7, s=60, edgecolors="black", linewidths=0.5,
)
ax.plot([shared_lim[0], shared_lim[1]], [shared_lim[0], shared_lim[1]], "k--", alpha=0.3, label="σ_block = σ_token")
ax.set_xlim(shared_lim)
ax.set_ylim(shared_lim)
ax.set_aspect("equal")
ax.set_xlabel(r"$\sigma_{\mathrm{token}}$", fontsize=13)
ax.set_ylabel(r"$\sigma_{\mathrm{block}}$", fontsize=13)
ax.set_title("(b) Non-causal texts", fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / _model_tagged_filename("scatter_token_vs_block.png"), dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / _model_tagged_filename("scatter_token_vs_block.pdf"), bbox_inches="tight")
if ARGS.show:
    plt.show()
plt.close(fig)
print(f"Figure saved in: {OUTPUT_DIR}")

# %% ── Cell 9: Concrete example of a causal text (log-likelihood comparison) ──

print("=" * 60)
print("Example: forward vs reversed log-likelihood profile for a causal text")
print("=" * 60)

# One illustrative causal text (selected by index, not by a representativeness criterion)
example_index = min(2, len(causal_texts) - 1)
example_text = causal_texts[example_index]
example_ids = tokenizer.encode(example_text, add_special_tokens=False)
example_tokens = [tokenizer.decode([tid]) for tid in example_ids]

sigma_example_full, _, _, per_tok_fwd, _, boundary_corr = compute_sigma_token(example_ids, model, device, tokenizer)

reversed_ids = list(reversed(example_ids))
_, per_tok_rev = compute_log_likelihood(reversed_ids, model, device, tokenizer)
reversed_tokens = [tokenizer.decode([tid]) for tid in reversed_ids]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

# Forward
ax = axes[0]
positions = range(len(per_tok_fwd))
bars = ax.bar(positions, per_tok_fwd, color="#2ECC71", alpha=0.7, edgecolor="black", linewidth=0.5)
ax.set_xticks(positions)
ax.set_xticklabels(example_tokens, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("log p(y_t | BOS, y_{1:t-1})", fontsize=12)
ax.set_title(f"Forward: log-prob per token (L_fwd = {sum(per_tok_fwd):.1f})", fontsize=13)
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
ax.grid(axis="y", alpha=0.3)

# Reversed
ax = axes[1]
bars = ax.bar(positions, per_tok_rev, color="#E74C3C", alpha=0.7, edgecolor="black", linewidth=0.5)
ax.set_xticks(positions)
ax.set_xticklabels(reversed_tokens, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("log p(z_t | BOS, z_{1:t-1})", fontsize=12)
ax.set_title(f"Token-reversed: log-prob per token (L_rev = {sum(per_tok_rev):.1f})", fontsize=13)
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / _model_tagged_filename("per_token_logprob_example.png"), dpi=150, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / _model_tagged_filename("per_token_logprob_example.pdf"), bbox_inches="tight")
if ARGS.show:
    plt.show()
plt.close(fig)

sigma_cond_only = sum(per_tok_fwd[1:]) - sum(per_tok_rev[1:])
print(f"\nText: {example_text}")
print(f"σ_cond = Σ_{{t≥2}} log p(y_t|BOS,y_{{1:t-1}}) - Σ_{{t≥2}} log p(z_t|BOS,z_{{1:t-1}}) = {sigma_cond_only:.1f}")
print(f"Boundary correction = ln p(y_1|BOS) - ln p(z_1|BOS) = {boundary_corr:.2f}")
print(f"σ_token (full) = σ_cond + boundary = {sigma_example_full:.1f}")

# %% ── Cell 10: Detailed analysis of block reversal ──

print("=" * 60)
print("Detailed analysis of block reversal")
print("=" * 60)

example_causal = causal_texts[0]
example_noncausal = noncausal_texts[0]

for label, text in [("Causal", example_causal), ("Non-causal", example_noncausal)]:
    print(f"\n{'─'*50}")
    print(f"[{label}]")
    print(f"Original text:  {text}")

    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Forward log-likelihood (computed once, shared by both functions)
    fwd_result = compute_full_log_likelihood(
        token_ids, model, device, tokenizer
    )

    # Compute sigma_block and use its returned text_blocks for display
    sigma_blk, ll_f, ll_br, text_blocks, rev_text_blocks, n_blocks = \
        compute_sigma_block(
            text, token_ids, model, device, tokenizer,
            _precomputed_fwd=fwd_result
        )

    print(f"Blocks ({n_blocks}):")
    for j, b in enumerate(text_blocks):
        print(f"  B{j+1}: \"{b}\"")

    rev_text = "".join(rev_text_blocks)
    print(f"Block-reversed actually evaluated: {rev_text!r}")

    # Compute sigma_token (reuse forward log-likelihood)
    sigma_tok, _, ll_tr, _, _, _ = compute_sigma_token(
        token_ids, model, device, tokenizer,
        _precomputed_fwd=fwd_result
    )

    print(f"\n  L_fwd          = {ll_f:.2f}")
    print(f"  L_token_rev    = {ll_tr:.2f}  →  σ_token = {sigma_tok:.2f}")
    print(f"  L_block_rev    = {ll_br:.2f}  →  σ_block = {sigma_blk:.2f}")
    if np.isclose(sigma_blk, 0.0, atol=1e-10, rtol=0.0):
        print("  σ_block ≈ 0")
    else:
        print(f"  σ_token / σ_block = {sigma_tok / sigma_blk:.2f}")

# %% ── Cell 11: Save figures, raw data, and reproducibility metadata ──

_OUT_FIXED = OUTPUT_DIR

# ── (1) Raw results CSV ──
_csv_path = os.path.join(_OUT_FIXED, _model_tagged_filename("raw_results_fixed_texts.csv"))
_fields = ["category","index","text","T","n_blocks","ll_fwd","ll_token_rev","ll_block_rev",
           "sigma_token","sigma_block","sigma_token_per_T","sigma_block_per_T"]
with open(_csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=_fields)
    w.writeheader()
    for r in results_causal:
        row = {k: r[k] for k in _fields if k != "category"}
        row["category"] = "causal"
        w.writerow(row)
    for r in results_noncausal:
        row = {k: r[k] for k in _fields if k != "category"}
        row["category"] = "noncausal"
        w.writerow(row)

# ── (2) Summary statistics CSV (fixed-text portion: 8 rows) ──
def _make_stat_row(label, values):
    arr = np.array(values)
    return {
        "condition": label, "N": len(arr),
        "mean": arr.mean(), "std": arr.std(ddof=1),
        "SE": arr.std(ddof=1) / np.sqrt(len(arr)),
        "median": np.median(arr), "min": arr.min(), "max": arr.max(),
        "q25": np.percentile(arr, 25), "q75": np.percentile(arr, 75),
    }

_summary_fields = ["condition","N","mean","std","SE","median","min","max","q25","q75"]
_csv_summary = os.path.join(_OUT_FIXED, _model_tagged_filename("summary_statistics_fixed.csv"))
with open(_csv_summary, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=_summary_fields)
    w.writeheader()
    w.writerow(_make_stat_row("sigma_token__causal",          sigmas_tok_c))
    w.writerow(_make_stat_row("sigma_token__noncausal",       sigmas_tok_nc))
    w.writerow(_make_stat_row("sigma_block__causal",          sigmas_blk_c))
    w.writerow(_make_stat_row("sigma_block__noncausal",       sigmas_blk_nc))
    w.writerow(_make_stat_row("sigma_token_perT__causal",
                              [r["sigma_token_per_T"] for r in results_causal]))
    w.writerow(_make_stat_row("sigma_token_perT__noncausal",
                              [r["sigma_token_per_T"] for r in results_noncausal]))
    w.writerow(_make_stat_row("sigma_block_perT__causal",
                              [r["sigma_block_per_T"] for r in results_causal]))
    w.writerow(_make_stat_row("sigma_block_perT__noncausal",
                              [r["sigma_block_per_T"] for r in results_noncausal]))

# ── (3) Selected environment and provenance fingerprints ──

# (a) nvidia-smi
_nvidia_smi = "N/A"
try:
    _nv = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    _nvidia_smi = _nv.stdout
    with open(os.path.join(_OUT_FIXED, _model_tagged_filename("nvidia_smi.txt")), "w") as f:
        f.write(_nvidia_smi)
except Exception:
    pass

# (b) cuDNN determinism flags
_determinism = {
    "cudnn_deterministic": torch.backends.cudnn.deterministic if torch.cuda.is_available() else "N/A",
    "cudnn_benchmark":     torch.backends.cudnn.benchmark     if torch.cuda.is_available() else "N/A",
    "cudnn_version":       str(torch.backends.cudnn.version()) if torch.cuda.is_available() else "N/A",
}

# (c) Model state-dict fingerprint (SHA-256 over sorted keys and tensor bytes)
_hasher = hashlib.sha256()
for _k, _v in sorted(model.state_dict().items()):
    _hasher.update(_k.encode())
    _hasher.update(_v.cpu().numpy().tobytes())
_model_hash = _hasher.hexdigest()

# (d) Tokenizer vocabulary-only fingerprint (does not include BPE merges/config)
_tok_hasher = hashlib.sha256()
for _token, _id in sorted(tokenizer.get_vocab().items()):
    _tok_hasher.update(f"{_token}:{_id}".encode())
_tokenizer_hash = _tok_hasher.hexdigest()

# (e) Best-effort OS release snapshot (normally available on Linux-like systems)
_os_release = "N/A"
try:
    with open("/etc/os-release") as f:
        _os_release = f.read()
    with open(os.path.join(_OUT_FIXED, _model_tagged_filename("os_release.txt")), "w") as f:
        f.write(_os_release)
except Exception:
    pass

# (f) Best-effort pip freeze snapshot for the active Python environment
try:
    _pip = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                         capture_output=True, text=True, timeout=30)
    with open(os.path.join(_OUT_FIXED, _model_tagged_filename("pip_freeze.txt")), "w", encoding="utf-8") as f:
        f.write(_pip.stdout)
except Exception:
    pass

# ── (4) Selected experiment metadata JSON ──
try:
    import transformers as _tf
    _tf_ver = _tf.__version__
except Exception:
    _tf_ver = "unknown"

# These fields describe detected CUDA/GPU availability.  The separately stored
# device_used field identifies whether this run actually evaluated on CUDA or CPU.
_gpu_info = "N/A (CPU only)"
_cuda_version = "N/A"
if torch.cuda.is_available():
    _gpu_info = torch.cuda.get_device_name(0)
    try:
        _gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        _gpu_info += f" ({_gpu_mem:.1f} GB)"
    except Exception:
        pass
    # torch.version.cuda is the CUDA version associated with the PyTorch build,
    # not the NVIDIA driver version reported by nvidia-smi.
    _cuda_version = torch.version.cuda or "N/A"

# timestamp_utc below is the metadata-creation time, after numerical analysis.
# runtime is a fixed label in this script rather than an auto-detected frontend.
_metadata_fixed = {
    "experiment": "Fixed-text entropy production (GPT-2)",
    "reference_paper": "Sagawa (2026), Stochastic Thermodynamics for Autoregressive Generative Models: A Non-Markovian Perspective",
    "input_text_metadata": texts_data.get("metadata", {}),
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "random_seed": RANDOM_SEED,
    "environment": {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": _cuda_version,
        "transformers_version": _tf_ver,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "gpu": _gpu_info,
        "device_used": str(device),
        "runtime": "local_python",
        "cudnn_deterministic": _determinism["cudnn_deterministic"],
        "cudnn_benchmark": _determinism["cudnn_benchmark"],
        "cudnn_version": _determinism["cudnn_version"],
    },
    "model": {
        "name_or_path": ARGS.model,
        "source": "huggingface/transformers or local directory",
        "vocab_size": tokenizer.vocab_size,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "state_dict_sha256": _model_hash,
        "tokenizer_vocab_sha256": _tokenizer_hash,
    },
    "experiment_config": {
        "n_causal_texts": len(causal_texts),
        "n_noncausal_texts": len(noncausal_texts),
        "texts_source": texts_source,
        "texts_local_path": str(texts_path),
        "texts_file_sha256": _texts_file_sha256,
        "texts_canonical_sha256": _texts_canonical_sha256,
        "local_files_only": ARGS.local_files_only,
        "add_prefix_space": True,
        "expected_blocks_per_text": ARGS.expected_blocks,
        # Semantic shorthand: the implementation actually detects any decoded
        # token whose rstripped final character is one of '.', '!', or '?'.
        "block_delimiter": "sentence-final punctuation (. ! ?)",
    },
    # These are means of unnormalized sigma.  Figure 4 uses the per-token values
    # stored in the model-tagged raw-results and summary-statistics CSV files.
    "key_results": {
        "sigma_token_mean_causal": float(np.mean(sigmas_tok_c)),
        "sigma_token_mean_noncausal": float(np.mean(sigmas_tok_nc)),
        "sigma_block_mean_causal": float(np.mean(sigmas_blk_c)),
        "sigma_block_mean_noncausal": float(np.mean(sigmas_blk_nc)),
    },
}
with open(os.path.join(_OUT_FIXED, _model_tagged_filename("experiment_metadata.json")), "w", encoding="utf-8") as f:
    json.dump(_metadata_fixed, f, indent=2, ensure_ascii=False)

# ── (5) Figures were saved directly into the output directory ──

# ── (6) Create a ZIP from the allowlisted files currently present ──
# Because a nonempty output directory is permitted, a preexisting optional file
# can be included if the corresponding best-effort collection failed this run.
_zip_path = Path(f"{_OUT_FIXED}_{_TEXT_SET_MODEL_TAG}.zip")
_generated_names = [_model_tagged_filename(name) for name in [
    "raw_results_fixed_texts.csv",
    "summary_statistics_fixed.csv",
    "experiment_metadata.json",
    "entropy_production_comparison.png",
    "entropy_production_comparison.pdf",
    "scatter_token_vs_block.png",
    "scatter_token_vs_block.pdf",
    "per_token_logprob_example.png",
    "per_token_logprob_example.pdf",
    "nvidia_smi.txt",
    "os_release.txt",
    "pip_freeze.txt",
]]
with zipfile.ZipFile(_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
    for name in _generated_names:
        path = _OUT_FIXED / name
        if path.is_file():
            zf.write(path, arcname=name)
print(f"[Saved] output directory: {_OUT_FIXED}")
print(f"[Saved] ZIP archive: {_zip_path}")

# %% [markdown]
# ## Notes
#
# - See the paper for theoretical background.
# - Generated result tables, figures, and selected environment metadata are
#   saved in `--output-dir`.
# - The model-tagged `experiment_metadata_*.json` file records selected
#   provenance details; it is not a standalone, complete reproduction recipe.
