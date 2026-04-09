# =============================================================================
# Entropy Production for Autoregressive Models: GPT-2 Numerical Experiment
# =============================================================================
# Paper: "Stochastic Thermodynamics for Autoregressive Generative Models"
# Estimation of entropy production — proof-of-concept demonstration
#
# Can be pasted directly into Google Colab and run as-is.
# Cell boundaries are marked with "# %%"; paste each cell separately in Colab.
# =============================================================================

# %% [markdown]
# # Numerical Demonstration of Entropy Production (GPT-2)
#
# **Comparison axes:**
# 1. Token reversal vs block (sentence) reversal
# 2. Causal texts vs non-causal texts
#
# **Theory:**
# $$\sigma(y_{1:T}) = \ln P(y_{1:T}) - \ln P(\text{reversed sequence})$$
#
# - Token reversal: reverse the order of all tokens
# - Block reversal: reverse the order of sentence blocks, preserving token order within each block

# %% ── Cell 1: Setup ──
!pip install transformers torch matplotlib japanize-matplotlib -q

import torch
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size':        16,   
    'axes.titlesize':   18,   
    'axes.labelsize':   16,  
    'xtick.labelsize':  13,   
    'ytick.labelsize':  13,  
    'legend.fontsize':  14,   
    'figure.titlesize': 20,  
})
import matplotlib
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Japanese font support (optional, for displaying Japanese in Colab)
try:
    import japanize_matplotlib
except ImportError:
    pass

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Set and record random seed (for reproducibility)
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f"Random seed: {RANDOM_SEED}")

# Load GPT-2
print("Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
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
    Compute the full log-likelihood of a token sequence,
    including the initial-distribution term for the first token.

    BOS (<|endoftext|>) is prepended internally to match GPT-2's
    generation convention.  The model receives [BOS, y_1, ..., y_T],
    and this function computes

        L = Σ_{t=1}^{T} log p(y_t | BOS, y_{1:t-1})

    Returns:
        total_ll: total log-likelihood (scalar)
        per_token_ll: per-token log-likelihood
                      (array of length T; element 0 = log p(y_1 | BOS))
    """
    # +1 for the prepended BOS token
    if len(token_ids) + 1 > model.config.n_positions:
        raise ValueError(
            f"Token count ({len(token_ids)}) + BOS exceeds GPT-2 context length "
            f"({model.config.n_positions}). Results would be invalid."
        )
    bos_id = tokenizer.eos_token_id  # GPT-2: <|endoftext|> serves as BOS
    input_seq = [bos_id] + list(token_ids)       # [BOS, y_1, ..., y_T]
    input_ids = torch.tensor([input_seq], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        # logits shape: (T+1, vocab_size)
        # logits[0] -> predictive distribution for y_1 given BOS
        # logits[t] -> predictive distribution for y_{t+1} given BOS, y_{1:t}
        logits = outputs.logits[0]

    # Convert to log-probabilities via log_softmax
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Extract log p(y_{t+1} | BOS, y_{1:t}) = log_probs[t, token_ids[t]]
    # for t = 0, ..., T-1  (i.e. predictions at positions 0 through T-1)
    per_token_ll = []
    for t in range(len(token_ids)):
        ll = log_probs[t, token_ids[t]].item()
        per_token_ll.append(ll)

    per_token_ll = np.array(per_token_ll)
    total_ll = per_token_ll.sum()

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

    Both forward and reversed sequences are evaluated with BOS prepended:
      forward:  model receives [BOS, y_1, y_2, ..., y_T]
      reversed: model receives [BOS, y_T, y_{T-1}, ..., y_1]
    This is handled internally by compute_full_log_likelihood.

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
    This avoids BPE re-tokenization issues and guarantees the bijection condition.
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
    This guarantees the bijection condition.

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

import json
import urllib.request

# Download text data from the GitHub repository
TEXTS_URL = "https://raw.githubusercontent.com/taksagawa/Paper2026/main/gpt2/text_sets/texts_gpt_exp_Opus1.json"

print("Downloading text data...")
texts_data = None
try:
    with urllib.request.urlopen(TEXTS_URL) as response:
        texts_data = json.loads(response.read().decode("utf-8"))
    print(f"  Downloaded from GitHub.")
except Exception as e:
    print(f"  GitHub download failed ({e}).")
    print("  Falling back to local file upload...")
    # In Colab: prompt the user to upload the JSON file
    try:
        from google.colab import files
        uploaded = files.upload()  # user selects texts_gpt_exp_Opus1.json
        for filename, content in uploaded.items():
            texts_data = json.loads(content.decode("utf-8"))
            print(f"  Loaded from uploaded file: {filename}")
            break
    except ImportError:
        # Not in Colab: try loading from the current directory
        import os
        local_path = "texts_gpt_exp_Opus1.json"
        if os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                texts_data = json.load(f)
            print(f"  Loaded from local file: {local_path}")

if texts_data is None:
    raise RuntimeError(
        "Could not load text data. Please either:\n"
        f"  (1) Push texts_gpt_exp_Opus1.json to {TEXTS_URL}, or\n"
        "  (2) Upload the file when prompted, or\n"
        "  (3) Place the file in the current working directory."
    )

causal_texts = texts_data["causal_texts"]
noncausal_texts = texts_data["noncausal_texts"]

assert len(causal_texts) == 30, f"Expected 30 causal texts, got {len(causal_texts)}"
assert len(noncausal_texts) == 30, f"Expected 30 non-causal texts, got {len(noncausal_texts)}"

print(f"Causal texts: {len(causal_texts)} samples")
print(f"Non-causal texts: {len(noncausal_texts)} samples")

# %% ── Cell 4: Main computation ──

def analyze_texts(texts, label, model, device, tokenizer):
    """Compute sigma_token and sigma_block for a set of texts."""
    results = []

    for i, text in enumerate(texts):
        token_ids = tokenizer.encode(text)
        T = len(token_ids)

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
single_ids = tokenizer.encode(single_text)
sigma_blk_single, _, _, _, _, _ = compute_sigma_block(
    single_text, single_ids, model, device, tokenizer
)
print(f"  σ_block = {sigma_blk_single:.6f}  (expected: 0.0)")

# Check 2: bijection condition verification
# Paper: block reversal R_S must be a bijection on Y^T
# Since we operate at the token level, |reversed| == |original|
print("\n[Check 2] Bijection check: |reversed_ids| == |original_ids|")
test_text2 = "She opened her eyes. She got out of bed. She made coffee."
test_ids2 = tokenizer.encode(test_text2)
token_blocks = split_token_ids_into_blocks(test_ids2, tokenizer)
reversed_token_ids_2 = []
for block in reversed(token_blocks):
    reversed_token_ids_2.extend(block)
bijection_ok = len(reversed_token_ids_2) == len(test_ids2)
# Also verify that double reversal recovers the original
blocks_of_reversed = split_token_ids_into_blocks(reversed_token_ids_2, tokenizer)
double_reversed = []
for block in reversed(blocks_of_reversed):
    double_reversed.extend(block)
round_trip_ok = double_reversed == test_ids2
print(f"  |original| = {len(test_ids2)}, |reversed| = {len(reversed_token_ids_2)} → {'OK ✓' if bijection_ok else 'FAIL ✗'}")
print(f"  Round-trip (reverse twice = identity): {'OK ✓' if round_trip_ok else 'FAIL ✗'}")

# Check 3: non-negativity of E[sigma] (sample mean)
sigmas_tok_c = [r["sigma_token"] for r in results_causal]
sigmas_tok_nc = [r["sigma_token"] for r in results_noncausal]
sigmas_blk_c = [r["sigma_block"] for r in results_causal]
sigmas_blk_nc = [r["sigma_block"] for r in results_noncausal]

print("\n[Check 3] Sample mean of σ")
print(f"  E[σ_token] (causal)     = {np.mean(sigmas_tok_c):.2f} ≥ 0 ✓" if np.mean(sigmas_tok_c) >= 0 else f"  E[σ_token] (causal)     = {np.mean(sigmas_tok_c):.2f} ✗")
print(f"  E[σ_token] (non-causal) = {np.mean(sigmas_tok_nc):.2f} ≥ 0 ✓" if np.mean(sigmas_tok_nc) >= 0 else f"  E[σ_token] (non-causal) = {np.mean(sigmas_tok_nc):.2f} ✗")
print(f"  E[σ_block] (causal)     = {np.mean(sigmas_blk_c):.2f} ≥ 0 ✓" if np.mean(sigmas_blk_c) >= 0 else f"  E[σ_block] (causal)     = {np.mean(sigmas_blk_c):.2f} ✗")
print(f"  E[σ_block] (non-causal) = {np.mean(sigmas_blk_nc):.2f} ≥ 0 ✓" if np.mean(sigmas_blk_nc) >= 0 else f"  E[σ_block] (non-causal) = {np.mean(sigmas_blk_nc):.2f} ✗")

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

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
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
plt.savefig("entropy_production_comparison.png", dpi=150, bbox_inches="tight")
plt.savefig("entropy_production_comparison.pdf", bbox_inches="tight")
plt.show()
print("Figure saved: entropy_production_comparison.png / .pdf")

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
plt.savefig("scatter_token_vs_block.png", dpi=150, bbox_inches="tight")
plt.savefig("scatter_token_vs_block.pdf", bbox_inches="tight")
plt.show()
print("Figure saved: scatter_token_vs_block.png / .pdf")

# %% ── Cell 9: Concrete example of a causal text (log-likelihood comparison) ──

print("=" * 60)
print("Example: forward vs reversed log-likelihood profile for a causal text")
print("=" * 60)

# One representative causal text
example_text = causal_texts[2]  # "The ball was thrown..."
example_ids = tokenizer.encode(example_text)
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
plt.savefig("per_token_logprob_example.png", dpi=150, bbox_inches="tight")
plt.savefig("per_token_logprob_example.pdf", bbox_inches="tight")
plt.show()

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

    token_ids = tokenizer.encode(text)

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

    rev_text = " ".join(rev_text_blocks)
    print(f"Block-reversed: {rev_text}")

    # Compute sigma_token (reuse forward log-likelihood)
    sigma_tok, _, ll_tr, _, _, _ = compute_sigma_token(
        token_ids, model, device, tokenizer,
        _precomputed_fwd=fwd_result
    )

    print(f"\n  L_fwd          = {ll_f:.2f}")
    print(f"  L_token_rev    = {ll_tr:.2f}  →  σ_token = {sigma_tok:.2f}")
    print(f"  L_block_rev    = {ll_br:.2f}  →  σ_block = {sigma_blk:.2f}")
    print(f"  σ_token / σ_block = {sigma_tok / sigma_blk:.2f}" if sigma_blk != 0 else "  σ_block = 0")

# %% ── Cell 11: Download fixed-text figures and raw data ──

import csv, json, os, shutil, platform, datetime, sys, hashlib, subprocess

_OUT_FIXED = "fixed_text_output"
os.makedirs(_OUT_FIXED, exist_ok=True)

# ── (1) Raw results CSV ──
_csv_path = os.path.join(_OUT_FIXED, "raw_results_fixed_texts.csv")
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
_csv_summary = os.path.join(_OUT_FIXED, "summary_statistics_fixed.csv")
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

# ── (3) Reproducibility fingerprints ──

# (a) nvidia-smi
_nvidia_smi = "N/A"
try:
    _nv = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    _nvidia_smi = _nv.stdout
    with open(os.path.join(_OUT_FIXED, "nvidia_smi.txt"), "w") as f:
        f.write(_nvidia_smi)
except Exception:
    pass

# (b) cuDNN determinism flags
_determinism = {
    "cudnn_deterministic": torch.backends.cudnn.deterministic if torch.cuda.is_available() else "N/A",
    "cudnn_benchmark":     torch.backends.cudnn.benchmark     if torch.cuda.is_available() else "N/A",
    "cudnn_version":       str(torch.backends.cudnn.version()) if torch.cuda.is_available() else "N/A",
}

# (c) Model weight hash (SHA-256)
_hasher = hashlib.sha256()
for _k, _v in sorted(model.state_dict().items()):
    _hasher.update(_k.encode())
    _hasher.update(_v.cpu().numpy().tobytes())
_model_hash = _hasher.hexdigest()

# (d) Tokenizer vocab hash (SHA-256)
_tok_hasher = hashlib.sha256()
for _token, _id in sorted(tokenizer.get_vocab().items()):
    _tok_hasher.update(f"{_token}:{_id}".encode())
_tokenizer_hash = _tok_hasher.hexdigest()

# (e) OS release (Colab system image)
_colab_image = "N/A"
try:
    with open("/etc/os-release") as f:
        _colab_image = f.read()
    with open(os.path.join(_OUT_FIXED, "os_release.txt"), "w") as f:
        f.write(_colab_image)
except Exception:
    pass

# (f) pip freeze
try:
    _pip = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                         capture_output=True, text=True, timeout=30)
    with open(os.path.join(_OUT_FIXED, "pip_freeze.txt"), "w", encoding="utf-8") as f:
        f.write(_pip.stdout)
except Exception:
    pass

# ── (4) Experiment metadata JSON ──
try:
    import transformers as _tf
    _tf_ver = _tf.__version__
except Exception:
    _tf_ver = "unknown"

_gpu_info = "N/A (CPU only)"
_cuda_version = "N/A"
if torch.cuda.is_available():
    _gpu_info = torch.cuda.get_device_name(0)
    try:
        _gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        _gpu_info += f" ({_gpu_mem:.1f} GB)"
    except Exception:
        pass
    _cuda_version = torch.version.cuda or "N/A"

_metadata_fixed = {
    "experiment": "Fixed-text entropy production (GPT-2)",
    "reference_paper": "Sagawa (2026), Stochastic Thermodynamics for Autoregressive Generative Models: A Non-Markovian Perspective",
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
        "is_google_colab": "google.colab" in sys.modules,
        "cudnn_deterministic": _determinism["cudnn_deterministic"],
        "cudnn_benchmark": _determinism["cudnn_benchmark"],
        "cudnn_version": _determinism["cudnn_version"],
    },
    "model": {
        "name": "gpt2",
        "source": "huggingface/transformers",
        "vocab_size": tokenizer.vocab_size,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "state_dict_sha256": _model_hash,
        "tokenizer_vocab_sha256": _tokenizer_hash,
    },
    "experiment_config": {
        "n_causal_texts": len(causal_texts),
        "n_noncausal_texts": len(noncausal_texts),
        "texts_source_url": TEXTS_URL,
        "block_delimiter": "sentence-final punctuation (. ! ?)",
    },
    "key_results": {
        "sigma_token_mean_causal": float(np.mean(sigmas_tok_c)),
        "sigma_token_mean_noncausal": float(np.mean(sigmas_tok_nc)),
        "sigma_block_mean_causal": float(np.mean(sigmas_blk_c)),
        "sigma_block_mean_noncausal": float(np.mean(sigmas_blk_nc)),
    },
}
with open(os.path.join(_OUT_FIXED, "experiment_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(_metadata_fixed, f, indent=2, ensure_ascii=False)

# ── (5) Copy figures ──
for fig in ["entropy_production_comparison.png", "entropy_production_comparison.pdf",
            "scatter_token_vs_block.png", "scatter_token_vs_block.pdf",
            "per_token_logprob_example.png", "per_token_logprob_example.pdf"]:
    if os.path.exists(fig):
        shutil.copy2(fig, os.path.join(_OUT_FIXED, fig))

# ── (6) Zip and download ──
shutil.make_archive(_OUT_FIXED, "zip", _OUT_FIXED)
print(f"[Saved] {_OUT_FIXED}.zip")
try:
    from google.colab import files
    files.download(f"{_OUT_FIXED}.zip")
    print("Download triggered (fixed-text results).")
except ImportError:
    print(f"Not in Colab — ZIP: {os.path.abspath(_OUT_FIXED)}.zip")

# %% [markdown]
# ## Notes
#
# - See the paper for theoretical background.
# - All raw data, figures, and environment metadata are saved in `fixed_text_output/`.
# - See `experiment_metadata.json` for full reproducibility information.