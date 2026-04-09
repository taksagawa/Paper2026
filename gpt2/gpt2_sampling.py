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
# # Monte Carlo Estimation of Entropy Production (GPT-2)
#
# **Method:** Sample y_{1:T} ~ P_fwd from GPT-2, compute \sigma for each sample.
#
# $$\hat{\sigma} = \frac{1}{N}\sum_{i=1}^{N} \sigma(y^{(i)}_{1:T})$$
#
# - Token reversal and block (sentence) reversal
# - See the companion notebook (`gpt2_input_text`) for the fixed-text experiment.


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

def split_token_ids_into_blocks(token_ids):
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


# %% ── Cell 3: Adjustable parameters ──
# =====================================================================
# All parameters are centrally managed here.
# =====================================================================

MAX_NEW_TOKENS = 120   # max generated tokens (truncated at punctuation for block reversal)
N_SAMPLES      = 500   # number of samples (block reversal target; token reversal may yield more)
TEMPERATURE    = 1.0   # sampling temperature

print("=" * 60)
print("  Monte Carlo Estimation — Parameters")
print("=" * 60)
print(f"  Max new tokens : {MAX_NEW_TOKENS}")
print(f"  N samples      : {N_SAMPLES}")
print(f"  Temperature    : τ = {TEMPERATURE}")
print("=" * 60)

# Sanity check: token length must fit in GPT-2 context window (1024)
assert MAX_NEW_TOKENS + 1 <= model.config.n_positions, \
    f"MAX_NEW_TOKENS={MAX_NEW_TOKENS} + BOS exceeds GPT-2 context length ({model.config.n_positions})"


# ── Temperature-aware log-likelihood ──

def compute_log_likelihood_kvcache(token_ids, model, device, tokenizer,
                                   temperature):
    """
    Log-likelihood at temperature τ, using the KV-cache code path.

    Computes  Σ_{t=1}^{T} ln q_τ(y_t | BOS, y_{1:t-1})
    where  q_τ(· | context) = softmax(logits(context) / τ),
    by feeding one token at a time with KV-cache — the identical
    computation path used during sampling in generate_and_compute_sigma().

    This ensures that forward (sampling) and reverse (evaluation)
    log-likelihoods traverse exactly the same floating-point operations,
    eliminating any numerical discrepancy between KV-cache and
    full-sequence forward passes.
    """
    if len(token_ids) + 1 > model.config.n_positions:
        raise ValueError(
            f"Token count ({len(token_ids)}) + BOS exceeds GPT-2 context "
            f"length ({model.config.n_positions})."
        )
    bos_id = tokenizer.eos_token_id
    past_key_values = None
    next_input = torch.tensor([[bos_id]], dtype=torch.long).to(device)

    per_token_ll = []
    for t in range(len(token_ids)):
        with torch.no_grad():
            outputs = model(next_input,
                            past_key_values=past_key_values,
                            use_cache=True)
        logits_t = outputs.logits[0, -1, :]        # (vocab_size,)
        past_key_values = outputs.past_key_values

        lp_t = torch.nn.functional.log_softmax(
            logits_t / temperature, dim=-1
        )
        per_token_ll.append(lp_t[token_ids[t]].item())

        next_input = torch.tensor([[token_ids[t]]],
                                  dtype=torch.long).to(device)

    per_token_ll = np.array(per_token_ll)
    return per_token_ll.sum(), per_token_ll


# %% ── Cell 4: GPT-2 sample generation + Monte Carlo estimation ──

print()
print("=" * 60)
print("  Monte Carlo estimation from GPT-2-generated text")
print("=" * 60)
print("(Approximately y_{1:T} ~ P_fwd sampling)")
print()

def generate_and_compute_sigma(model, tokenizer, device, prompt,
                               max_new_tokens=120, n_samples=300,
                               temperature=1.0):
    """
    Generate text from GPT-2 and compute sigma_token and sigma_block for each sample.
    This approximately corresponds to the Monte Carlo estimator in eq. (sigma-MC).

    Sampling is performed via a manual KV-cache loop (NOT model.generate)
    so that the exact per-token log-probabilities can be recorded from the
    SAME logits used for sampling.  Reverse log-likelihoods are also
    evaluated via the same KV-cache code path, so that forward and reverse
    traverse identical floating-point operations.

    Every sample has exactly T = max_new_tokens tokens.

    Token reversal uses the full generated sequence (length T).
    Block reversal requires the bijection condition (y_T must be a delimiter),
    so generated sequences are truncated at the last sentence-final punctuation
    token; sigma_block is computed on this truncated sequence (length T').
    Samples with no punctuation are valid for token reversal but skipped
    for block reversal.  As a result, the number of token-reversal samples
    may exceed that of block-reversal samples.
    The loop continues until n_samples block-reversal results are collected.

    Additionally, for each sample with a valid T', sigma_token is also
    computed on the truncated sequence y_{1:T'} (sigma_token_Tprime) as
    a reference quantity for comparison with sigma_block.
    """
    results_gen = []
    n_valid_blk = 0   # counter for samples with valid block reversal

    attempts = 0
    max_attempts = n_samples * 5  # prevent infinite loop

    # Loop until block reversal (the more restrictive condition) reaches
    # n_samples.  Token reversal is always valid (len == max_new_tokens),
    # so its count will be >= n_valid_blk.
    while n_valid_blk < n_samples and attempts < max_attempts:
        attempts += 1
        # ── Sample tokens with KV-cache, recording exact log-probs ──
        # Both forward and reverse log-likelihoods use the same KV-cache
        # code path, ensuring identical floating-point operations.
        if prompt == "":
            init_ids = [tokenizer.eos_token_id]
        else:
            init_ids = tokenizer.encode(prompt)
        token_ids = []
        log_probs_fwd = []
        past_key_values = None
        next_input = torch.tensor([init_ids], dtype=torch.long).to(device)

        for _t in range(max_new_tokens):
            with torch.no_grad():
                _out = model(next_input,
                             past_key_values=past_key_values,
                             use_cache=True)
            _logits_t = _out.logits[0, -1, :]
            past_key_values = _out.past_key_values

            _lp_t = torch.nn.functional.log_softmax(
                _logits_t / temperature, dim=-1
            )
            _probs_t = torch.exp(_lp_t)
            _next_tok = torch.multinomial(_probs_t, num_samples=1).item()

            token_ids.append(_next_tok)
            log_probs_fwd.append(_lp_t[_next_tok].item())

            next_input = torch.tensor([[_next_tok]],
                                      dtype=torch.long).to(device)

        # ── Token reversal: always use the full generated sequence ──
        token_ids_tok = list(token_ids)
        T_tok = len(token_ids_tok)
        text = tokenizer.decode(token_ids_tok, skip_special_tokens=True)

        # Forward log-likelihood = exact sum of sampling log-probs
        ll_fwd = sum(log_probs_fwd)

        # σ_token at temperature τ (always computed)
        reversed_ids_tok = list(reversed(token_ids_tok))
        ll_rev, _ = compute_log_likelihood_kvcache(
            reversed_ids_tok, model, device, tokenizer, temperature
        )
        sigma_tok = ll_fwd - ll_rev

        # ── Block reversal: truncate at the last punctuation token ──
        blk_valid = True
        if token_ids[-1] not in PUNCT_TOKEN_IDS:
            last_punct = -1
            for k in range(len(token_ids) - 1, -1, -1):
                if token_ids[k] in PUNCT_TOKEN_IDS:
                    last_punct = k
                    break
            if last_punct < 1:
                blk_valid = False  # no punctuation found
            else:
                token_ids_blk = token_ids[:last_punct + 1]
                if len(token_ids_blk) < 2:
                    blk_valid = False  # truncated sequence too short
        else:
            token_ids_blk = token_ids_tok  # no truncation needed

        if blk_valid:
            T_blk = len(token_ids_blk)
            # Forward LL for truncated sequence = first T' sampling log-probs
            ll_fwd_blk = sum(log_probs_fwd[:T_blk])

            # σ_block at temperature τ
            token_blocks = split_token_ids_into_blocks(token_ids_blk)
            n_blocks = len(token_blocks)
            if n_blocks <= 1:
                sigma_blk = 0.0
            else:
                reversed_block_ids = []
                for _block in reversed(token_blocks):
                    reversed_block_ids.extend(_block)
                ll_rev_blk, _ = compute_log_likelihood_kvcache(
                    reversed_block_ids, model, device, tokenizer, temperature
                )
                sigma_blk = ll_fwd_blk - ll_rev_blk

            # σ_token on the truncated sequence y_{1:T'} (reference)
            reversed_ids_Tp = list(reversed(token_ids_blk))
            ll_rev_Tp, _ = compute_log_likelihood_kvcache(
                reversed_ids_Tp, model, device, tokenizer, temperature
            )
            sigma_tok_Tp = ll_fwd_blk - ll_rev_Tp

            n_valid_blk += 1
        else:
            token_ids_blk = None
            T_blk = None
            sigma_blk = None
            sigma_tok_Tp = None
            n_blocks = None

        results_gen.append({
            "token_ids": token_ids_tok,
            "token_ids_blk": token_ids_blk,
            "text": text,
            "T": T_tok,
            "T_blk": T_blk,
            "n_blocks": n_blocks,
            "sigma_token": sigma_tok,
            "sigma_block": sigma_blk,
            "sigma_token_Tprime": sigma_tok_Tp,
            "sigma_token_per_T": sigma_tok / T_tok,
            "sigma_block_per_Tprime": (sigma_blk / T_blk) if blk_valid else None,
            "sigma_token_Tprime_per_Tprime": (sigma_tok_Tp / T_blk) if blk_valid else None,
        })

        i = len(results_gen) - 1
        if i < 3:
            if blk_valid:
                print(f"  Sample {i}: T={T_tok}, T'={T_blk}, blocks={n_blocks}, "
                      f"σ_tok/T={sigma_tok/T_tok:.2f}, σ_blk/T'={sigma_blk/T_blk:.3f}, "
                      f"σ_tok(T')/T'={sigma_tok_Tp/T_blk:.3f}")
            else:
                print(f"  Sample {i}: T={T_tok}, "
                      f"σ_tok/T={sigma_tok/T_tok:.2f}, block reversal: N/A (no delimiter)")
            print(f"    {text[:100]}...")

    n_tok = len(results_gen)
    n_blk = sum(1 for r in results_gen if r["sigma_block"] is not None)
    n_skipped_blk = n_tok - n_blk

    print(f"\n  Total attempts: {attempts}")
    print(f"  Token reversal: {n_tok} valid samples")
    print(f"  Block reversal: {n_blk} valid samples ({n_skipped_blk} skipped — no delimiter)")

    # Token reversal arrays (all samples)
    sigmas_tok = np.array([r["sigma_token"] for r in results_gen])
    token_lens_tok = np.array([r["T"] for r in results_gen])
    sigmas_tok_per_T = sigmas_tok / token_lens_tok

    # Block reversal arrays (only valid samples)
    sigmas_blk = np.array([r["sigma_block"] for r in results_gen
                           if r["sigma_block"] is not None])
    token_lens_blk = np.array([r["T_blk"] for r in results_gen
                               if r["T_blk"] is not None])
    sigmas_blk_per_T = sigmas_blk / token_lens_blk

    # Token reversal at T' arrays (only valid block-reversal samples)
    sigmas_tok_Tp = np.array([r["sigma_token_Tprime"] for r in results_gen
                              if r["sigma_token_Tprime"] is not None])
    sigmas_tok_Tp_per_T = sigmas_tok_Tp / token_lens_blk

    se_tok = sigmas_tok.std(ddof=1) / np.sqrt(len(sigmas_tok))
    se_blk = sigmas_blk.std(ddof=1) / np.sqrt(len(sigmas_blk))
    se_tok_Tp = sigmas_tok_Tp_per_T.std(ddof=1) / np.sqrt(len(sigmas_tok_Tp_per_T))

    # Compute block-reversal statistics only for samples with >= 2 blocks
    multi_block_mask = np.array([r["n_blocks"] is not None and r["n_blocks"] >= 2
                                 for r in results_gen
                                 if r["sigma_block"] is not None])
    n_multi = multi_block_mask.sum()

    print(f"\n  Monte Carlo estimates:")
    print(f"    E[σ_token]    = {sigmas_tok.mean():.2f} ± {se_tok:.2f} (SE)  [N={n_tok}]")
    print(f"    E[σ_token/T]  = {sigmas_tok_per_T.mean():.4f} ± {sigmas_tok_per_T.std(ddof=1)/np.sqrt(len(sigmas_tok_per_T)):.4f} (SE)")
    print(f"    E[σ_block]    = {sigmas_blk.mean():.2f} ± {se_blk:.2f} (SE)  [N={n_blk}]")
    print(f"    E[σ_block/T'] = {sigmas_blk_per_T.mean():.4f} ± {sigmas_blk_per_T.std(ddof=1)/np.sqrt(len(sigmas_blk_per_T)):.4f} (SE)")
    print(f"    E[σ_token(T')/T'] = {sigmas_tok_Tp_per_T.mean():.4f} ± {se_tok_Tp:.4f} (SE)  [N={len(sigmas_tok_Tp_per_T)}] (ref)")
    print(f"    Samples with ≥ 2 blocks: {n_multi}/{n_blk}")
    if n_multi > 0:
        blk_multi = sigmas_blk_per_T[multi_block_mask]
        print(f"    E[σ_block/T'] (≥2 blocks only) = {blk_multi.mean():.4f} ± {blk_multi.std(ddof=1)/np.sqrt(len(blk_multi)):.4f} (SE)")
    print(f"    Fraction σ_token < 0: {np.mean(sigmas_tok < 0):.2%}")
    print(f"    Fraction σ_block < 0: {np.mean(sigmas_blk < 0):.2%}")

    return results_gen, attempts

# No prompt (start from <|endoftext|> as initial context)
print(f"\n--- Prompt: (empty / BOS), τ = {TEMPERATURE} ---")
results_generated, _total_attempts = generate_and_compute_sigma(
    model, tokenizer, device, prompt="",
    max_new_tokens=MAX_NEW_TOKENS, n_samples=N_SAMPLES,
    temperature=TEMPERATURE,
)

sigmas_generated = np.array([r["sigma_token"] for r in results_generated])
token_lengths_generated = np.array([r["T"] for r in results_generated])
sigmas_per_T_generated = sigmas_generated / token_lengths_generated

sigmas_blk_generated = np.array([r["sigma_block"] for r in results_generated
                                 if r["sigma_block"] is not None])
token_lengths_blk_generated = np.array([r["T_blk"] for r in results_generated
                                        if r["T_blk"] is not None])
sigmas_blk_per_T_generated = sigmas_blk_generated / token_lengths_blk_generated

# Token reversal at T' (reference: same truncated sequence as block reversal)
sigmas_tok_Tp_generated = np.array([r["sigma_token_Tprime"] for r in results_generated
                                    if r["sigma_token_Tprime"] is not None])
sigmas_tok_Tp_per_T_generated = sigmas_tok_Tp_generated / token_lengths_blk_generated

# Histogram (per-token entropy production: token reversal vs block reversal)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# (a) \sigma_token / T
ax = axes[0]
ax.hist(sigmas_per_T_generated, bins=20, color="#9B59B6", alpha=0.7,
        edgecolor="black", linewidth=0.5, density=True,
        label=f"$\\sigma_{{\\mathrm{{token}}}}/T$  (N={len(sigmas_per_T_generated)})")
ax.hist(sigmas_tok_Tp_per_T_generated, bins=20,
        histtype='step', color="#E67E22", linewidth=2.0, linestyle="--", density=True,
        label=f"Ref: $\\sigma_{{\\mathrm{{token}}}}(T')/T'$  (N={len(sigmas_tok_Tp_per_T_generated)})")
ax.axvline(x=sigmas_per_T_generated.mean(), color="red", linestyle="--",
           linewidth=2, label=f"Mean($\\sigma_{{\\mathrm{{token}}}}/T$) = {sigmas_per_T_generated.mean():.2f}")
ax.axvline(x=sigmas_tok_Tp_per_T_generated.mean(), color="#E67E22", linestyle=":",
           linewidth=2, label=f"Mean($\\sigma_{{\\mathrm{{token}}}}(T')/T'$) = {sigmas_tok_Tp_per_T_generated.mean():.2f}")
ax.axvline(x=0, color="gray", linestyle="-", linewidth=1)
ax.set_xlabel(r"$\sigma_{\mathrm{token}} / T$"
              r"$\;$or$\;$"
              r"$\sigma_{\mathrm{token}}(T') / T'$", fontsize=17)
ax.set_ylabel("Density", fontsize=18)
ax.set_title("(a) Token reversal", fontsize=20)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# (b) \sigma_block / T'
ax = axes[1]
ax.hist(sigmas_blk_per_T_generated, bins=20, color="#2ECC71", alpha=0.7,
        edgecolor="black", linewidth=0.5, density=True,
        label=f"N = {len(sigmas_blk_per_T_generated)} samples")
ax.axvline(x=sigmas_blk_per_T_generated.mean(), color="red", linestyle="--",
           linewidth=2, label=f"Mean = {sigmas_blk_per_T_generated.mean():.3f}")
ax.axvline(x=0, color="gray", linestyle="-", linewidth=1)
ax.set_xlabel(r"$\sigma_{\mathrm{block}} / T'$", fontsize=20)
ax.set_ylabel("Density", fontsize=18)
ax.set_title("(b) Block reversal", fontsize=20)
ax.legend(fontsize=14)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("sigma_distribution_generated.png", dpi=150, bbox_inches="tight")
plt.savefig("sigma_distribution_generated.pdf", bbox_inches="tight")
plt.show()

# %% ── Cell 4b: Convergence plot with Bootstrap CI ──

import csv as _csv_mod   # needed if this cell runs independently in Colab

print("\n" + "=" * 60)
print("  Convergence plot — Bootstrap 95% CI (percentile method)")
print("=" * 60)

B_BOOT = 2000   # number of bootstrap resamples
CONV_STEP = 10  # evaluate at N = 10, 20, 30, ...


def bootstrap_convergence(values, step, B, rng):
    """
    Cumulative mean and 95% bootstrap percentile CI at N = step, 2*step, ...,
    plus N_max if it is not a multiple of step.

    Parameters
    ----------
    values : 1-D array of per-sample quantities (e.g. sigma/T)
    step   : evaluation interval
    B      : number of bootstrap resamples
    rng    : np.random.RandomState for reproducibility

    Returns
    -------
    N_vals      : 1-D array of sample counts
    cum_means   : cumulative sample mean at each N
    ci_lo, ci_hi: 2.5-th and 97.5-th percentiles of bootstrap means
    """
    N_max = len(values)
    N_vals = np.arange(step, N_max + 1, step)
    if len(N_vals) == 0:
        N_vals = np.array([N_max])
    elif N_vals[-1] < N_max:
        N_vals = np.append(N_vals, N_max)

    cum_means = np.empty(len(N_vals))
    ci_lo     = np.empty(len(N_vals))
    ci_hi     = np.empty(len(N_vals))

    for idx, n in enumerate(N_vals):
        subset = values[:n]
        cum_means[idx] = subset.mean()
        # Bootstrap: draw (B, n) indices, compute row-wise means
        boot_idx = rng.randint(0, n, size=(B, n))
        boot_means = subset[boot_idx].mean(axis=1)
        ci_lo[idx] = np.percentile(boot_means, 2.5)
        ci_hi[idx] = np.percentile(boot_means, 97.5)

    return N_vals, cum_means, ci_lo, ci_hi


# ── Token reversal ──
print("  Computing bootstrap CI for σ_token/T ...")
rng_boot = np.random.RandomState(RANDOM_SEED)
N_tok_conv, mean_tok_conv, lo_tok_conv, hi_tok_conv = bootstrap_convergence(
    sigmas_per_T_generated, step=CONV_STEP, B=B_BOOT, rng=rng_boot,
)
print(f"    Done. {len(N_tok_conv)} points, "
      f"N = {N_tok_conv[0]}..{N_tok_conv[-1]}")

# ── Block reversal ──
print("  Computing bootstrap CI for σ_block/T' ...")
rng_boot = np.random.RandomState(RANDOM_SEED)
N_blk_conv, mean_blk_conv, lo_blk_conv, hi_blk_conv = bootstrap_convergence(
    sigmas_blk_per_T_generated, step=CONV_STEP, B=B_BOOT, rng=rng_boot,
)
print(f"    Done. {len(N_blk_conv)} points, "
      f"N = {N_blk_conv[0]}..{N_blk_conv[-1]}")

# ── Plot ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# (a) σ_token / T
ax = axes[0]
ax.fill_between(N_tok_conv, lo_tok_conv, hi_tok_conv,
                color="#3498DB", alpha=0.25, label="95% Bootstrap CI")
ax.plot(N_tok_conv, mean_tok_conv, color="#2C3E50", linewidth=1.5,
        label="Cumulative mean")
ax.axhline(y=mean_tok_conv[-1], color="red", linestyle="--", linewidth=1.2,
           label=f"Final estimate = {mean_tok_conv[-1]:.4f}")
ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
ax.set_xlabel("$N$ (number of samples)", fontsize=20)
ax.set_ylabel(r"$\langle \sigma_{\mathrm{token}} / T \rangle$", fontsize=18)
ax.set_title("(a) Token reversal", fontsize=20)
ax.legend(fontsize=14, loc="lower right")
ax.grid(alpha=0.3)

# (b) σ_block / T'
ax = axes[1]
ax.fill_between(N_blk_conv, lo_blk_conv, hi_blk_conv,
                color="#2ECC71", alpha=0.25, label="95% Bootstrap CI")
ax.plot(N_blk_conv, mean_blk_conv, color="#2C3E50", linewidth=1.5,
        label="Cumulative mean")
ax.axhline(y=mean_blk_conv[-1], color="red", linestyle="--", linewidth=1.2,
           label=f"Final estimate = {mean_blk_conv[-1]:.4f}")
ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
ax.set_xlabel("$N$ (number of samples)", fontsize=20)
ax.set_ylabel(r"$\langle \sigma_{\mathrm{block}} / T' \rangle$", fontsize=18)
ax.set_title("(b) Block reversal", fontsize=20)
ax.legend(fontsize=14, loc="lower right")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("sigma_convergence_generated.png", dpi=150, bbox_inches="tight")
plt.savefig("sigma_convergence_generated.pdf", bbox_inches="tight")
plt.show()
print("[Saved] sigma_convergence_generated.png")
print("[Saved] sigma_convergence_generated.pdf")

# ── Save convergence raw data CSV ──
_csv_conv_path = "convergence_bootstrap.csv"
with open(_csv_conv_path, "w", newline="", encoding="utf-8") as _f:
    _w = _csv_mod.writer(_f)
    _w.writerow(["reversal_type", "N", "cumulative_mean",
                 "ci_lower_2.5", "ci_upper_97.5"])
    for _i in range(len(N_tok_conv)):
        _w.writerow(["token", int(N_tok_conv[_i]),
                      mean_tok_conv[_i], lo_tok_conv[_i], hi_tok_conv[_i]])
    for _i in range(len(N_blk_conv)):
        _w.writerow(["block", int(N_blk_conv[_i]),
                      mean_blk_conv[_i], lo_blk_conv[_i], hi_blk_conv[_i]])
print(f"[Saved] {_csv_conv_path}")


# %% ── Cell 5: Download Monte Carlo figures and raw data ──

import csv, json, os, shutil, platform, datetime, sys, hashlib, subprocess

_OUT_MC = "monte_carlo_output"
os.makedirs(_OUT_MC, exist_ok=True)

# ── (1) Raw results CSV ──
_csv_path = os.path.join(_OUT_MC, "raw_results_generated_samples.csv")
_fields = ["sample_index","T","T_blk","n_blocks","sigma_token","sigma_block",
           "sigma_token_Tprime",
           "sigma_token_per_T","sigma_block_per_Tprime",
           "sigma_token_Tprime_per_Tprime","text","token_ids"]
with open(_csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=_fields)
    w.writeheader()
    for i, r in enumerate(results_generated):
        w.writerow({
            "sample_index": i, "T": r["T"],
            "T_blk": r["T_blk"] if r["T_blk"] is not None else "",
            "n_blocks": r["n_blocks"] if r["n_blocks"] is not None else "",
            "sigma_token": r["sigma_token"],
            "sigma_block": r["sigma_block"] if r["sigma_block"] is not None else "",
            "sigma_token_Tprime": r["sigma_token_Tprime"] if r["sigma_token_Tprime"] is not None else "",
            "sigma_token_per_T": r["sigma_token_per_T"],
            "sigma_block_per_Tprime": r["sigma_block_per_Tprime"] if r["sigma_block_per_Tprime"] is not None else "",
            "sigma_token_Tprime_per_Tprime": r["sigma_token_Tprime_per_Tprime"] if r["sigma_token_Tprime_per_Tprime"] is not None else "",
            "text": r["text"], "token_ids": json.dumps(r["token_ids"]),
        })

# ── (2) Summary statistics CSV (Monte Carlo portion: 6 rows) ──
def _make_stat_row_mc(label, values):
    arr = np.array(values)
    return {
        "condition": label, "N": len(arr),
        "mean": arr.mean(), "std": arr.std(ddof=1),
        "SE": arr.std(ddof=1) / np.sqrt(len(arr)),
        "median": np.median(arr), "min": arr.min(), "max": arr.max(),
        "q25": np.percentile(arr, 25), "q75": np.percentile(arr, 75),
    }

_summary_fields = ["condition","N","mean","std","SE","median","min","max","q25","q75"]
_csv_summary = os.path.join(_OUT_MC, "summary_statistics_montecarlo.csv")
with open(_csv_summary, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=_summary_fields)
    w.writeheader()
    w.writerow(_make_stat_row_mc("generated__sigma_token", sigmas_generated.tolist()))
    w.writerow(_make_stat_row_mc("generated__sigma_block", sigmas_blk_generated.tolist()))
    w.writerow(_make_stat_row_mc("generated__sigma_token_Tprime",
                                 sigmas_tok_Tp_generated.tolist()))
    w.writerow(_make_stat_row_mc("generated__sigma_token_perT",
                                 sigmas_per_T_generated.tolist()))
    w.writerow(_make_stat_row_mc("generated__sigma_block_perTprime",
                                 sigmas_blk_per_T_generated.tolist()))
    w.writerow(_make_stat_row_mc("generated__sigma_token_Tprime_perTprime",
                                 sigmas_tok_Tp_per_T_generated.tolist()))

# ── (3) Reproducibility fingerprints ──
#     Compute reproducibility fingerprints.

# (a) nvidia-smi
try:
    _nvidia_smi  
except NameError:
    _nvidia_smi = "N/A"
    try:
        _nv = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        _nvidia_smi = _nv.stdout
    except Exception:
        pass
if _nvidia_smi and _nvidia_smi != "N/A":
    with open(os.path.join(_OUT_MC, "nvidia_smi.txt"), "w") as f:
        f.write(_nvidia_smi)

# (b) cuDNN determinism flags
try:
    _determinism
except NameError:
    _determinism = {
        "cudnn_deterministic": torch.backends.cudnn.deterministic if torch.cuda.is_available() else "N/A",
        "cudnn_benchmark":     torch.backends.cudnn.benchmark     if torch.cuda.is_available() else "N/A",
        "cudnn_version":       str(torch.backends.cudnn.version()) if torch.cuda.is_available() else "N/A",
    }

# (c) Model weight hash
try:
    _model_hash
except NameError:
    _hasher = hashlib.sha256()
    for _k, _v in sorted(model.state_dict().items()):
        _hasher.update(_k.encode())
        _hasher.update(_v.cpu().numpy().tobytes())
    _model_hash = _hasher.hexdigest()

# (d) Tokenizer vocab hash
try:
    _tokenizer_hash
except NameError:
    _tok_hasher = hashlib.sha256()
    for _token, _id in sorted(tokenizer.get_vocab().items()):
        _tok_hasher.update(f"{_token}:{_id}".encode())
    _tokenizer_hash = _tok_hasher.hexdigest()

# (e) OS release
try:
    _colab_image
except NameError:
    _colab_image = "N/A"
    try:
        with open("/etc/os-release") as f:
            _colab_image = f.read()
    except Exception:
        pass
if _colab_image and _colab_image != "N/A":
    with open(os.path.join(_OUT_MC, "os_release.txt"), "w") as f:
        f.write(_colab_image)

# (f) pip freeze
try:
    _pip = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                         capture_output=True, text=True, timeout=30)
    with open(os.path.join(_OUT_MC, "pip_freeze.txt"), "w", encoding="utf-8") as f:
        f.write(_pip.stdout)
except Exception:
    pass

# ── (4) Experiment metadata JSON ──
try:
    _tf_ver
except NameError:
    try:
        import transformers as _tf
        _tf_ver = _tf.__version__
    except Exception:
        _tf_ver = "unknown"

_gpu_info_mc = "N/A (CPU only)"
_cuda_ver_mc = "N/A"
if torch.cuda.is_available():
    _gpu_info_mc = torch.cuda.get_device_name(0)
    try:
        _gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        _gpu_info_mc += f" ({_gpu_mem:.1f} GB)"
    except Exception:
        pass
    _cuda_ver_mc = torch.version.cuda or "N/A"

_metadata_mc = {
    "experiment": "Monte Carlo entropy production (GPT-2 generated text)",
    "reference_paper": "Sagawa (2026), Stochastic Thermodynamics for Autoregressive Generative Models: A Non-Markovian Perspective",
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "random_seed": RANDOM_SEED,
    "environment": {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": _cuda_ver_mc,
        "transformers_version": _tf_ver,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "gpu": _gpu_info_mc,
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
        "n_requested_samples": N_SAMPLES,
        "n_total_attempts": _total_attempts,
        "n_valid_token_reversal": len(results_generated),
        "n_valid_block_reversal": int(sum(1 for r in results_generated
                                          if r["sigma_block"] is not None)),
        "n_skipped_block_reversal": int(sum(1 for r in results_generated
                                            if r["sigma_block"] is None)),
        "block_skip_rate_no_delimiter": int(sum(1 for r in results_generated
                                               if r["sigma_block"] is None)) / _total_attempts,
        "generated_max_new_tokens": MAX_NEW_TOKENS,
        "generated_temperature": TEMPERATURE,
    },
    "key_results": {
        "generated_sigma_token_per_T_mean": float(sigmas_per_T_generated.mean()),
        "generated_sigma_block_per_Tprime_mean": float(sigmas_blk_per_T_generated.mean()),
        "generated_sigma_token_Tprime_per_Tprime_mean": float(sigmas_tok_Tp_per_T_generated.mean()),
        "generated_sigma_token_N": len(sigmas_generated),
        "generated_sigma_block_N": len(sigmas_blk_generated),
        "generated_sigma_token_Tprime_N": len(sigmas_tok_Tp_generated),
    },
}
with open(os.path.join(_OUT_MC, "experiment_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(_metadata_mc, f, indent=2, ensure_ascii=False)

# ── (5) Copy figures and convergence data ──
if os.path.exists("sigma_distribution_generated.png"):
    shutil.copy2("sigma_distribution_generated.png",
                 os.path.join(_OUT_MC, "sigma_distribution_generated.png"))
if os.path.exists("sigma_convergence_generated.png"):
    shutil.copy2("sigma_convergence_generated.png",
                 os.path.join(_OUT_MC, "sigma_convergence_generated.png"))
if os.path.exists("sigma_distribution_generated.pdf"):
    shutil.copy2("sigma_distribution_generated.pdf",
                 os.path.join(_OUT_MC, "sigma_distribution_generated.pdf"))
if os.path.exists("sigma_convergence_generated.pdf"):
    shutil.copy2("sigma_convergence_generated.pdf",
                 os.path.join(_OUT_MC, "sigma_convergence_generated.pdf"))
if os.path.exists("convergence_bootstrap.csv"):
    shutil.copy2("convergence_bootstrap.csv",
                 os.path.join(_OUT_MC, "convergence_bootstrap.csv"))

# ── (6) Zip and download ──
shutil.make_archive(_OUT_MC, "zip", _OUT_MC)
print(f"[Saved] {_OUT_MC}.zip")
try:
    from google.colab import files
    files.download(f"{_OUT_MC}.zip")
    print("Download triggered (Monte Carlo results).")
except ImportError:
    print(f"Not in Colab — ZIP: {os.path.abspath(_OUT_MC)}.zip")

# %% [markdown]
# ## Notes
#
# - See the paper for theoretical background.
# - All raw data, figures, and environment metadata are saved in `monte_carlo_output/`.
# - See `experiment_metadata.json` for full reproducibility information.
# - See the companion notebook (`gpt2_input_text`) for the fixed-text experiment.