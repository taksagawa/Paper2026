# =============================================================================
# Fluctuation Theorem Verification for Autoregressive Models (GPT-2)
# =============================================================================
# Verifies the fluctuation theorem:  <e^{-sigma_token}> = 1
# where sigma_token = ln P_sample(y_{1:T}) - ln Q_tau(y_{T:1})
#   P_sample: exact log-prob recorded during sampling (forward)
#   Q_tau:    model evaluation at temperature tau   (reverse)
#
# Two temperature settings are compared side by side.
#
# Based on: "Stochastic Thermodynamics for Autoregressive Generative Models"
#
# Can be pasted directly into Google Colab and run as-is.
# Cell boundaries are marked with "# %%"; paste each cell separately in Colab.
# =============================================================================

# %% [markdown]
# # Fluctuation Theorem Verification (GPT-2)
#
# **Goal:** Verify $\langle e^{-\sigma_{\mathrm{token}}} \rangle = 1$
# where $\sigma_{\mathrm{token}} = \ln P_{\mathrm{sample}}(y_{1:T}) - \ln Q_\tau(y_{T:1})$.
# Both forward and reverse log-likelihoods are computed via the same
# KV-cache code path, ensuring identical floating-point operations.
#
# - Token reversal only (no block reversal).
# - $\sigma_{\mathrm{token}}$ is NOT divided by $T$.
# - Two temperatures $\tau_1, \tau_2$ are compared side by side.
# - Convergence of $\langle e^{-\sigma_{\mathrm{token}}} \rangle$ is plotted
#   as a function of sample count $N$, with bootstrap percentile 95% CI.


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


# %% ── Cell 2: Core functions ──

def compute_log_likelihood_kvcache(token_ids, model, device, tokenizer,
                                   temperature):
    """
    Log-likelihood at temperature τ, using the KV-cache code path.

    Computes  Σ_{t=1}^{T} ln q_τ(y_t | BOS, y_{1:t-1})
    where  q_τ(· | context) = softmax(logits(context) / τ),
    by feeding one token at a time with KV-cache — the identical
    computation path used during sampling in run_experiment().

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


# %% ── Cell 3: Adjustable parameters ──
# =====================================================================
# All parameters are centrally managed here.
# =====================================================================

MAX_NEW_TOKENS  = 5   # number of generated tokens per sample (fixed T)
N_SAMPLES       = 5000   # number of Monte Carlo samples (per temperature)
TEMPERATURE_1   = 3.0   # first  sampling & evaluation temperature
TEMPERATURE_2   = 4.0   # second sampling & evaluation temperature
B_BOOT          = 2000  # number of bootstrap resamples for convergence CI
CONV_STEP       = 10    # convergence evaluation interval (N = 10, 20, ...)

print("=" * 60)
print("  Fluctuation Theorem Verification — Parameters")
print("=" * 60)
print(f"  Max new tokens  : {MAX_NEW_TOKENS}")
print(f"  N samples        : {N_SAMPLES} (per temperature)")
print(f"  Temperature τ₁   : {TEMPERATURE_1}")
print(f"  Temperature τ₂   : {TEMPERATURE_2}")
print(f"  Bootstrap B      : {B_BOOT}")
print(f"  Convergence step : {CONV_STEP}")
print("=" * 60)

# Sanity check: token length must fit in GPT-2 context window (1024)
assert MAX_NEW_TOKENS + 1 <= model.config.n_positions, \
    f"MAX_NEW_TOKENS={MAX_NEW_TOKENS} + BOS exceeds GPT-2 context length ({model.config.n_positions})"


# %% ── Cell 4: Sampling function + execution for two temperatures ──

def run_experiment(model, tokenizer, device,
                   temperature, n_samples, max_new_tokens):
    """
    Run the fluctuation theorem experiment at a single temperature.

    For each of n_samples Monte Carlo samples:
      1. Sample y_{1:T} one token at a time at temperature τ, recording
         the exact per-token log-probability from the SAME logits used
         for sampling.  This gives  ln P_sample(y_{1:T}).
      2. Compute reverse log-likelihood  ln Q_τ(y_{T:1})  by evaluating
         the reversed sequence through the model using the KV-cache
         code path — the same floating-point path as step 1.
      3. σ_token = ln P_sample(y_{1:T}) − ln Q_τ(y_{T:1})  (NOT / T).

    CRITICAL — why the KV-cache path is used for BOTH directions:
      A full-sequence forward pass and a KV-cache incremental pass are
      mathematically equivalent (both use causal masking), but they may
      differ at the float32 level due to different matrix-multiplication
      decompositions.  Using the KV-cache path for both forward and
      reverse guarantees that P and Q traverse exactly the same
      floating-point operations, so  ⟨e^{-σ}⟩ = 1  holds precisely.

    Returns
    -------
    results      : list of dicts (per-sample details)
    sigmas_token : 1-D np.array of σ_token values
    exp_neg_sigma: 1-D np.array of e^{-σ_token} values
    """
    bos_id = tokenizer.eos_token_id
    results = []

    for i in range(n_samples):
        # ── Step 1: Sample tokens with KV-cache, recording log-probs ──
        token_ids = []
        log_probs_fwd = []       # exact sampling log-probs
        past_key_values = None
        next_input = torch.tensor([[bos_id]], dtype=torch.long).to(device)

        for t in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(next_input,
                                past_key_values=past_key_values,
                                use_cache=True)
            logits_t = outputs.logits[0, -1, :]        # (vocab_size,)
            past_key_values = outputs.past_key_values

            # Temperature-scaled log-probabilities (same logits → same distrib)
            lp_t = torch.nn.functional.log_softmax(
                logits_t / temperature, dim=-1
            )
            probs_t = torch.exp(lp_t)
            next_token = torch.multinomial(probs_t, num_samples=1).item()

            token_ids.append(next_token)
            log_probs_fwd.append(lp_t[next_token].item())

            next_input = torch.tensor([[next_token]],
                                      dtype=torch.long).to(device)

        T = len(token_ids)
        text = tokenizer.decode(token_ids, skip_special_tokens=True)

        # Forward log-likelihood = exact sum of sampling log-probs
        ll_fwd = sum(log_probs_fwd)

        # ── Step 2: Reverse log-likelihood (KV-cache evaluation) ──
        reversed_ids = list(reversed(token_ids))
        ll_rev, _ = compute_log_likelihood_kvcache(
            reversed_ids, model, device, tokenizer, temperature
        )

        # ── Step 3: σ_token (NOT divided by T) ──
        sigma_tok = ll_fwd - ll_rev

        results.append({
            "token_ids": token_ids,
            "text": text,
            "T": T,
            "sigma_token": sigma_tok,
            "ll_fwd": ll_fwd,
            "ll_rev": ll_rev,
        })

        # Progress reporting
        if i < 3:
            print(f"  Sample {i}: T={T}, σ_token={sigma_tok:.2f}, "
                  f"e^{{-σ}}={np.exp(-sigma_tok):.4e}")
            print(f"    {text[:100]}...")
        elif i % 50 == 0:
            sigmas_so_far = np.array([r["sigma_token"] for r in results])
            exp_neg_so_far = np.exp(-sigmas_so_far)
            print(f"  Sample {i}: ⟨e^{{-σ}}⟩_{{N={i+1}}} = "
                  f"{exp_neg_so_far.mean():.4e}")

    # ── Extract arrays ──
    sigmas_token = np.array([r["sigma_token"] for r in results])
    exp_neg_sigma = np.exp(-sigmas_token)

    print(f"\n  Complete: {len(results)} samples at τ = {temperature}")
    print(f"  ⟨σ_token⟩     = {sigmas_token.mean():.2f} "
          f"± {sigmas_token.std(ddof=1)/np.sqrt(len(sigmas_token)):.2f} (SE)")
    print(f"  ⟨e^{{-σ_token}}⟩ = {exp_neg_sigma.mean():.4e} "
          f"± {exp_neg_sigma.std(ddof=1)/np.sqrt(len(exp_neg_sigma)):.4e} (SE)")
    print(f"  Theoretical prediction: ⟨e^{{-σ_token}}⟩ = 1")
    print(f"  Fraction σ_token < 0: {np.mean(sigmas_token < 0):.2%}")
    print(f"  min(σ_token) = {sigmas_token.min():.2f}, "
          f"max(σ_token) = {sigmas_token.max():.2f}")
    print(f"  max(e^{{-σ_token}}) = {exp_neg_sigma.max():.4e}")

    return results, sigmas_token, exp_neg_sigma


# ── Run experiments for both temperatures ──
TEMPERATURES = [TEMPERATURE_1, TEMPERATURE_2]

experiments = {}   # keyed by temperature value

for tau in TEMPERATURES:
    # Reset RNG so that each temperature starts from the same seed,
    # making results independent of the other temperature's sample count.
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    print()
    print("=" * 60)
    print(f"  Experiment: τ = {tau}")
    print("=" * 60)
    print()

    results, sigmas, exp_neg = run_experiment(
        model, tokenizer, device,
        temperature=tau,
        n_samples=N_SAMPLES,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    experiments[tau] = {
        "results": results,
        "sigmas_token": sigmas,
        "exp_neg_sigma": exp_neg,
    }


# %% ── Cell 5: Convergence plot — ⟨e^{-σ_token}⟩ vs N (two panels) ──

import csv as _csv_mod   # needed if this cell runs independently in Colab

print("\n" + "=" * 60)
print("  Convergence plot — ⟨e^{-σ_token}⟩ vs N")
print("  Bootstrap 95% CI (percentile method)")
print("=" * 60)


def bootstrap_convergence_ft(values, step, B, rng):
    """
    Cumulative mean of e^{-sigma} and 95% bootstrap percentile CI
    at N = step, 2*step, ..., plus N_max if not a multiple of step.

    Parameters
    ----------
    values : 1-D array of σ_token for each sample
    step   : evaluation interval
    B      : number of bootstrap resamples
    rng    : np.random.RandomState for reproducibility

    Returns
    -------
    N_vals      : 1-D array of sample counts
    cum_means   : cumulative mean of e^{-σ} at each N
    ci_lo, ci_hi: 2.5-th and 97.5-th percentiles of bootstrap means
    """
    exp_neg = np.exp(-values)    # e^{-σ_token} for each sample
    N_max = len(values)
    if N_max == 0:
        raise ValueError("values must be non-empty")
    N_vals = np.arange(step, N_max + 1, step)
    if len(N_vals) == 0:
        # step > N_max: evaluate only at N_max
        N_vals = np.array([N_max])
    elif N_vals[-1] < N_max:
        N_vals = np.append(N_vals, N_max)

    cum_means = np.empty(len(N_vals))
    ci_lo     = np.empty(len(N_vals))
    ci_hi     = np.empty(len(N_vals))

    for idx, n in enumerate(N_vals):
        subset = exp_neg[:n]
        cum_means[idx] = subset.mean()
        # Bootstrap: draw (B, n) indices, compute row-wise means
        boot_idx = rng.randint(0, n, size=(B, n))
        boot_means = subset[boot_idx].mean(axis=1)
        ci_lo[idx] = np.percentile(boot_means, 2.5)
        ci_hi[idx] = np.percentile(boot_means, 97.5)

    return N_vals, cum_means, ci_lo, ci_hi


# ── Compute bootstrap CI for each temperature ──
convergence = {}   # keyed by temperature

for tau in TEMPERATURES:
    # Reset bootstrap RNG for each temperature independently.
    rng_boot = np.random.RandomState(RANDOM_SEED)
    print(f"  Computing bootstrap CI for τ = {tau} ...")
    N_conv, mean_conv, lo_conv, hi_conv = bootstrap_convergence_ft(
        experiments[tau]["sigmas_token"],
        step=CONV_STEP, B=B_BOOT, rng=rng_boot,
    )
    convergence[tau] = {
        "N": N_conv, "mean": mean_conv, "lo": lo_conv, "hi": hi_conv,
    }
    print(f"    Done. {len(N_conv)} points, "
          f"N = {N_conv[0]}..{N_conv[-1]}")

# ── Plot: two panels side by side ──
colors = ["#3498DB", "#E67E22"]   # blue for τ₁, orange for τ₂

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, tau, color in zip(axes, TEMPERATURES, colors):
    cv = convergence[tau]
    ax.fill_between(cv["N"], cv["lo"], cv["hi"],
                    color=color, alpha=0.25, label="95% Bootstrap CI")
    ax.plot(cv["N"], cv["mean"], color="#2C3E50", linewidth=1.5,
            label=r"Cumulative mean")
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5,
               label=r"Theoretical $= 1$")
    ax.set_xlabel("$N$ (number of samples)", fontsize=20)
    ax.set_ylabel(r"$\langle e^{-\sigma_{\mathrm{token}}} \rangle$",
                  fontsize=18)
    ax.set_title(rf"$\tau = {tau}$", fontsize=20)
    ax.legend(fontsize=14, loc="best")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fluctuation_theorem_convergence.png", dpi=150, bbox_inches="tight")
plt.savefig("fluctuation_theorem_convergence.pdf", bbox_inches="tight")
plt.show()
print("[Saved] fluctuation_theorem_convergence.png")
print("[Saved] fluctuation_theorem_convergence.pdf")

# ── Histogram 1: distribution of σ_token ──
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

for ax, tau, color in zip(axes2, TEMPERATURES, colors):
    sigmas = experiments[tau]["sigmas_token"]
    ax.hist(sigmas, bins=30, color=color, alpha=0.7,
            edgecolor="black", linewidth=0.5, density=True)
    ax.axvline(x=sigmas.mean(), color="red", linestyle="--", linewidth=1.5,
               label=rf"Mean $= {sigmas.mean():.1f}$")
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.8)
    ax.set_xlabel(r"$\sigma_{\mathrm{token}}$", fontsize=18)
    ax.set_ylabel("Density", fontsize=18)
    ax.set_title(rf"$\tau = {tau}$", fontsize=20)
    ax.legend(fontsize=12, loc="best")
    ax.grid(alpha=0.3)

fig2.suptitle(
    r"Distribution of $\sigma_{\mathrm{token}}$",
    fontsize=22, y=1.02,
)
plt.tight_layout()
plt.savefig("sigma_token_hist.png", dpi=150, bbox_inches="tight")
plt.savefig("sigma_token_hist.pdf", bbox_inches="tight")
plt.show()
print("[Saved] sigma_token_hist.png")
print("[Saved] sigma_token_hist.pdf")

# ── Histogram 2: distribution of e^{-σ_token} ──
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))

for ax, tau, color in zip(axes3, TEMPERATURES, colors):
    exp_neg = experiments[tau]["exp_neg_sigma"]
    ax.hist(exp_neg, bins=30, color=color, alpha=0.7,
            edgecolor="black", linewidth=0.5, density=True)
    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5,
               label=r"Theoretical $= 1$")
    ax.set_xlabel(r"$e^{-\sigma_{\mathrm{token}}}$", fontsize=18)
    ax.set_ylabel("Density", fontsize=18)
    ax.set_title(rf"$\tau = {tau}$", fontsize=20)
    ax.legend(fontsize=12, loc="best")
    ax.grid(alpha=0.3)

fig3.suptitle(
    r"Distribution of $e^{-\sigma_{\mathrm{token}}}$",
    fontsize=22, y=1.02,
)
plt.tight_layout()
plt.savefig("exp_neg_sigma_hist.png", dpi=150, bbox_inches="tight")
plt.savefig("exp_neg_sigma_hist.pdf", bbox_inches="tight")
plt.show()
print("[Saved] exp_neg_sigma_hist.png")
print("[Saved] exp_neg_sigma_hist.pdf")

# ── Save convergence raw data CSV (both temperatures) ──
_csv_conv_path = "convergence_ft_bootstrap.csv"
with open(_csv_conv_path, "w", newline="", encoding="utf-8") as _f:
    _w = _csv_mod.writer(_f)
    _w.writerow(["temperature", "N", "cumulative_mean_exp_neg_sigma",
                 "ci_lower_2.5", "ci_upper_97.5"])
    for tau in TEMPERATURES:
        cv = convergence[tau]
        for _i in range(len(cv["N"])):
            _w.writerow([tau, int(cv["N"][_i]),
                          cv["mean"][_i], cv["lo"][_i], cv["hi"][_i]])
print(f"[Saved] {_csv_conv_path}")


# %% ── Cell 6: Save results ──

import csv, json, os, shutil, platform, datetime, sys, hashlib, subprocess

_OUT = "fluctuation_theorem_output"
os.makedirs(_OUT, exist_ok=True)

# ── (1) Raw results CSV (both temperatures in one file) ──
_csv_path = os.path.join(_OUT, "raw_results.csv")
_fields = ["temperature", "sample_index", "T", "sigma_token",
           "exp_neg_sigma", "ll_fwd", "ll_rev", "text", "token_ids"]
with open(_csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=_fields)
    w.writeheader()
    for tau in TEMPERATURES:
        for i, r in enumerate(experiments[tau]["results"]):
            w.writerow({
                "temperature": tau,
                "sample_index": i, "T": r["T"],
                "sigma_token": r["sigma_token"],
                "exp_neg_sigma": np.exp(-r["sigma_token"]),
                "ll_fwd": r["ll_fwd"],
                "ll_rev": r["ll_rev"],
                "text": r["text"],
                "token_ids": json.dumps(r["token_ids"]),
            })

# ── (2) Summary statistics CSV (both temperatures) ──
def _make_stat_row(label, tau, values):
    arr = np.array(values)
    return {
        "quantity": label, "temperature": tau, "N": len(arr),
        "mean": arr.mean(), "std": arr.std(ddof=1),
        "SE": arr.std(ddof=1) / np.sqrt(len(arr)),
        "median": np.median(arr), "min": arr.min(), "max": arr.max(),
        "q25": np.percentile(arr, 25), "q75": np.percentile(arr, 75),
    }

_summary_fields = ["quantity", "temperature", "N", "mean", "std", "SE",
                    "median", "min", "max", "q25", "q75"]
_csv_summary = os.path.join(_OUT, "summary_statistics.csv")
with open(_csv_summary, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=_summary_fields)
    w.writeheader()
    for tau in TEMPERATURES:
        w.writerow(_make_stat_row(
            "sigma_token", tau,
            experiments[tau]["sigmas_token"].tolist()))
        w.writerow(_make_stat_row(
            "exp_neg_sigma_token", tau,
            experiments[tau]["exp_neg_sigma"].tolist()))

# ── (3) Reproducibility fingerprints ──

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
    with open(os.path.join(_OUT, "nvidia_smi.txt"), "w") as f:
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
    with open(os.path.join(_OUT, "os_release.txt"), "w") as f:
        f.write(_colab_image)

# (f) pip freeze
try:
    _pip = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                         capture_output=True, text=True, timeout=30)
    with open(os.path.join(_OUT, "pip_freeze.txt"), "w", encoding="utf-8") as f:
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

_gpu_info = "N/A (CPU only)"
_cuda_ver = "N/A"
if torch.cuda.is_available():
    _gpu_info = torch.cuda.get_device_name(0)
    try:
        _gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        _gpu_info += f" ({_gpu_mem:.1f} GB)"
    except Exception:
        pass
    _cuda_ver = torch.version.cuda or "N/A"

# Build per-temperature key_results
_key_results = {}
for tau in TEMPERATURES:
    s = experiments[tau]["sigmas_token"]
    e = experiments[tau]["exp_neg_sigma"]
    _key_results[f"tau={tau}"] = {
        "sigma_token_mean": float(s.mean()),
        "sigma_token_SE": float(s.std(ddof=1) / np.sqrt(len(s))),
        "exp_neg_sigma_mean": float(e.mean()),
        "exp_neg_sigma_SE": float(e.std(ddof=1) / np.sqrt(len(e))),
        "N": len(s),
    }

_metadata = {
    "experiment": "Fluctuation theorem verification (GPT-2, token reversal, "
                  "two temperatures)",
    "reference_paper": "Sagawa (2026), Stochastic Thermodynamics for "
                       "Autoregressive Generative Models: "
                       "A Non-Markovian Perspective",
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "random_seed": RANDOM_SEED,
    "environment": {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": _cuda_ver,
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
        "n_samples_per_temperature": N_SAMPLES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature_1": TEMPERATURE_1,
        "temperature_2": TEMPERATURE_2,
        "bootstrap_B": B_BOOT,
        "convergence_step": CONV_STEP,
    },
    "key_results": _key_results,
    "theoretical_prediction": 1.0,
}
with open(os.path.join(_OUT, "experiment_metadata.json"), "w",
          encoding="utf-8") as f:
    json.dump(_metadata, f, indent=2, ensure_ascii=False)

# ── (5) Copy figures and convergence data ──
if os.path.exists("fluctuation_theorem_convergence.png"):
    shutil.copy2("fluctuation_theorem_convergence.png",
                 os.path.join(_OUT, "fluctuation_theorem_convergence.png"))
if os.path.exists("fluctuation_theorem_convergence.pdf"):
    shutil.copy2("fluctuation_theorem_convergence.pdf",
                 os.path.join(_OUT, "fluctuation_theorem_convergence.pdf"))
if os.path.exists("convergence_ft_bootstrap.csv"):
    shutil.copy2("convergence_ft_bootstrap.csv",
                 os.path.join(_OUT, "convergence_ft_bootstrap.csv"))
if os.path.exists("sigma_token_hist.png"):
    shutil.copy2("sigma_token_hist.png",
                 os.path.join(_OUT, "sigma_token_hist.png"))
if os.path.exists("sigma_token_hist.pdf"):
    shutil.copy2("sigma_token_hist.pdf",
                 os.path.join(_OUT, "sigma_token_hist.pdf"))
if os.path.exists("exp_neg_sigma_hist.png"):
    shutil.copy2("exp_neg_sigma_hist.png",
                 os.path.join(_OUT, "exp_neg_sigma_hist.png"))
if os.path.exists("exp_neg_sigma_hist.pdf"):
    shutil.copy2("exp_neg_sigma_hist.pdf",
                 os.path.join(_OUT, "exp_neg_sigma_hist.pdf"))

# ── (6) Zip and download ──
shutil.make_archive(_OUT, "zip", _OUT)
print(f"[Saved] {_OUT}.zip")
try:
    from google.colab import files
    files.download(f"{_OUT}.zip")
    print("Download triggered (fluctuation theorem results).")
except ImportError:
    print(f"Not in Colab — ZIP: {os.path.abspath(_OUT)}.zip")

# %% [markdown]
# ## Notes
#
# - **Fluctuation theorem:** $\langle e^{-\sigma_{\mathrm{token}}} \rangle = 1$
#   where $\sigma_{\mathrm{token}} = \ln P_{\mathrm{sample}}(y_{1:T}) - \ln Q_\tau(y_{T:1})$.
#   The forward log-likelihood $\ln P_{\mathrm{sample}}$ is recorded from the exact
#   logits used during sampling, and the reverse log-likelihood $\ln Q_\tau$ is
#   evaluated using the same KV-cache code path, so that both directions
#   traverse identical floating-point operations.
# - $\sigma_{\mathrm{token}}$ is NOT divided by $T$.
# - The SAME temperature $\tau$ is used for both sampling and
#   log-likelihood evaluation.  This is required for the theorem to hold.
# - Convergence can be slow because rare samples with $\sigma < 0$ produce
#   very large $e^{-\sigma}$ values, dominating the average.
# - See the paper for theoretical background.
# - All raw data, figures, and metadata are saved in `fluctuation_theorem_output/`.
