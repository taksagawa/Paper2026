#!/usr/bin/env python3
"""
Numerical verification of the analytical entropy production S_y
for the linear Gaussian (Kalman innovation representation) case.

Analytical formula:  Eq. (eq:new-kl-R)
    S_y = (1/2) [ tr( (I_T ⊗ S^{-1}) R (I_T ⊗ S) R^T ) - T n_y ]
where R = H^{-1} J H  is the innovation reversal matrix.

Monte Carlo estimator:  Eq. (eq:sigma-per-traj) + Eq. (eq:sigma-MC)
    sigma(y_{1:T}) = sum_{t=0}^{T-1} ln p_t(y_{t+1} | f_t^{->}(y_{1:t}))
                   - sum_{t=1}^{T}   ln p_t(y_t   | g_{t+1}^{<-}(y_{T:t+1}))
    S_y ≈ (1/N) sum_n sigma(y_{1:T}^{(n)})

Reference:
    "Stochastic Thermodynamics for Autoregressive Generative Models"
    Section 6 (Linear Gaussian case: Kalman innovation representation)
"""

import numpy as np
from scipy.linalg import solve_discrete_are
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv


# =====================================================================
#  Steady-state Kalman filter quantities
# =====================================================================

def solve_kalman_steady_state(A, C, Q, R):
    """
    Solve the discrete algebraic Riccati equation (DARE) for the
    steady-state prediction error covariance P, and compute
    the innovation covariance S and the Kalman gain K.

    DARE:  P = A (P - P C^T (C P C^T + R)^{-1} C P) A^T + Q

    Returns
    -------
    P : ndarray (nx, nx)   prediction error covariance
    S : ndarray (ny, ny)   innovation covariance  S = C P C^T + R
    K : ndarray (nx, ny)   Kalman gain  K = P C^T S^{-1}
    """
    # scipy.linalg.solve_discrete_are(F, G, Qd, Rd) solves
    #   X = F^T X F - F^T X G (G^T X G + Rd)^{-1} G^T X F + Qd
    # Setting F = A^T, G = C^T recovers the standard Kalman DARE.
    P = solve_discrete_are(A.T, C.T, Q, R)
    S = C @ P @ C.T + R
    K = P @ C.T @ np.linalg.inv(S)
    return P, S, K


# =====================================================================
#  Analytical entropy production  (eq:new-kl-R)
# =====================================================================

def analytic_entropy_production(A, C, Q, R, T):
    """
    Compute S_y analytically via (eq:new-kl-R):

        S_y = (1/2) [ tr( (I_T ⊗ S^{-1}) R (I_T ⊗ S) R^T ) - T n_y ]

    where the innovation reversal matrix is R = H^{-1} J H  (eq:new-R-def),
    H is the block lower-triangular impulse-response matrix (eq:MA-rep),
    and J is the time-reversal permutation matrix (eq:J_def).
    """
    ny = C.shape[0]
    P, S, K = solve_kalman_steady_state(A, C, Q, R)

    # --- Build impulse-response coefficients H_l  (eq:H-coeff) ---
    #   H_0 = I_{ny},   H_l = C A^l K   (l >= 1)
    H_coeffs = [np.eye(ny)]
    Al = np.eye(A.shape[0])
    for l in range(1, T):
        Al = Al @ A
        H_coeffs.append(C @ Al @ K)

    # --- Build block lower-triangular matrix H ---
    #   H[i,j] = H_{i-j}  for i >= j,  else 0
    dim = T * ny
    Hmat = np.zeros((dim, dim))
    for i in range(T):
        for j in range(i + 1):
            Hmat[i*ny:(i+1)*ny, j*ny:(j+1)*ny] = H_coeffs[i - j]

    # --- Build time-reversal permutation matrix J  (eq:J_def) ---
    Jmat = np.zeros((dim, dim))
    for i in range(T):
        Jmat[i*ny:(i+1)*ny, (T-1-i)*ny:(T-i)*ny] = np.eye(ny)

    # --- Innovation reversal matrix  R = H^{-1} J H  (eq:new-R-def) ---
    Rmat = np.linalg.solve(Hmat, Jmat @ Hmat)

    # --- Entropy production  (eq:new-kl-R) ---
    Sinv = np.linalg.inv(S)
    IT_Sinv = np.kron(np.eye(T), Sinv)
    IT_S    = np.kron(np.eye(T), S)
    trace_val = np.trace(IT_Sinv @ Rmat @ IT_S @ Rmat.T)
    Sy = 0.5 * (trace_val - T * ny)
    return Sy


# =====================================================================
#  Monte Carlo estimation  (eq:sigma-per-traj, eq:sigma-MC)
#  ---- vectorized over N samples for performance ----
# =====================================================================

def mc_entropy_production(A, C, Q, R, T, N, rng):
    """
    Estimate S_y by Monte Carlo (eq:sigma-MC), vectorized over N samples.

    For each sampled trajectory y_{1:T} ~ P_{->}:
      1. Forward pass:  compute f_t^{->}(y_{1:t})  and forward innovations  e_t
      2. Backward pass:  compute g_{t+1}^{<-}(y_{T:t+1})  and backward innovations e_s^B
      3. Stochastic entropy production  (eq:sigma-per-traj):
           sigma = sum_{t=0}^{T-1} ln p_t(y_{t+1} | f_t^{->}(y_{1:t}))
                 - sum_{t=1}^{T}   ln p_t(y_t   | g_{t+1}^{<-}(y_{T:t+1}))

    In the Kalman case all emission kernels share the same N(Ch, S) form,
    so the Gaussian normalization constants cancel and we obtain:
           sigma = (1/2) sum_{s=1}^{T} (e_s^B)^T S^{-1} e_s^B
                 - (1/2) sum_{t=1}^{T} e_t^T S^{-1} e_t

    Note on ranges (eq:sigma-per-traj):
        Forward sum:  t = 0, ..., T-1  -->  T terms  (ln p_t(y_{t+1} | f_t^{->}))
        Backward sum: t = 1, ..., T    -->  T terms  (ln p_t(y_t   | g_{t+1}^{<-}))
        Both have exactly T terms; their normalization constants are identical and cancel.

    Returns
    -------
    mean_sigma : float     Monte Carlo mean of sigma
    se_sigma   : float     standard error  = std(sigma) / sqrt(N)
    sigmas     : ndarray   raw samples  sigma^{(n)},  shape (N,)
    """
    ny = C.shape[0]
    nx = A.shape[0]
    P, S, K = solve_kalman_steady_state(A, C, Q, R)
    Sinv   = np.linalg.inv(S)
    S_chol = np.linalg.cholesky(S)

    # Pre-compute matrix for the Kalman one-step map:
    #   f_{t+1}^{->}(y_{1:t+1}) = A [ f_t^{->}(y_{1:t}) + K e_{t+1} ]
    #                            = A f_t^{->}(y_{1:t}) + A K e_{t+1}
    # NOTE: When the input is the innovation e (not the observation y),
    #       the correct transition is simply A (not A_cl = A(I-KC)).
    AK = A @ K

    # --- Forward pass (vectorized over N) ---
    # Generate all innovations at once:  e_fwd[t] ~ N(0, S)  for t = 0,...,T-1
    # (e_fwd[t] corresponds to e_{t+1} in paper notation)
    noise = rng.standard_normal((T, N, ny))
    e_fwd = np.einsum("ij,tnj->tni", S_chol, noise)       # (T, N, ny)

    # Forward latent state:  h = f_t^{->}(y_{1:t})
    # h[t=0] = f_0^{->}(empty) = hat{x}_{1|0} = 0
    # Observation: y_{t+1} = C f_t^{->}(y_{1:t}) + e_{t+1}         (eq:gen_y)
    # Update:      f_{t+1}^{->}(y_{1:t+1}) = A h_t + A K e_{t+1}   (eq:update, eq:predict)
    y_traj = np.zeros((T, N, ny))
    h = np.zeros((N, nx))           # f_0^{->}(empty) = 0
    for t in range(T):
        y_traj[t] = h @ C.T + e_fwd[t]                    # y_{t+1}
        h = h @ A.T + e_fwd[t] @ AK.T                     # f_{t+1}^{->}

    # --- Backward pass (vectorized over N) ---
    # Feed the time-reversed sequence: tilde{y}_{s+1} = y_{T-s}
    # Backward latent state:  h_bwd = g_{T+1}^{<-}(empty) = hat{x}^B_{1|0} = 0
    e_bwd = np.zeros((T, N, ny))
    h = np.zeros((N, nx))           # g_{T+1}^{<-}(empty) = 0
    for s in range(T):
        # tilde{y}_{s+1} = y_{T-s}  -->  0-based: y_traj[T-1-s]
        tilde_y = y_traj[T - 1 - s]
        # Backward innovation:  e_{s+1}^B = tilde{y}_{s+1} - C g_{...}^{<-}
        e_b = tilde_y - h @ C.T
        e_bwd[s] = e_b
        # Backward Kalman update (same mechanism as forward)
        h = h @ A.T + e_b @ AK.T

    # --- Compute sigma per trajectory  (eq:sigma-per-traj) ---
    # sigma^{(n)} = (1/2) [ sum_s (e_s^B)^T S^{-1} e_s^B  -  sum_t e_t^T S^{-1} e_t ]
    fwd_quad = np.einsum("tni,ij,tnj->n", e_fwd, Sinv, e_fwd)   # (N,)
    bwd_quad = np.einsum("tni,ij,tnj->n", e_bwd, Sinv, e_bwd)   # (N,)
    sigmas = 0.5 * (bwd_quad - fwd_quad)

    mean_sigma = sigmas.mean()
    se_sigma   = sigmas.std(ddof=1) / np.sqrt(N)
    return mean_sigma, se_sigma, sigmas


# =====================================================================
#  Main: parameter sets, sweep over T, plot, and save
# =====================================================================

def main():
    # --- Parameters ---
    T_analytic = np.arange(1, 51)        # T = 1, 2, ..., 50  (analytic curve)
    T_mc       = np.arange(5, 51, 5)     # T = 5, 10, ..., 50 (Monte Carlo)
    N_mc       = 20_000                   # Monte Carlo sample size
    seed     = 42
    rng      = np.random.default_rng(seed)

    # --- Case 1: Scalar  n_x = n_y = 1 ---
    # S_y remains bounded as T -> infty  (boundary effect only; Weiss 1975)
    A1 = np.array([[0.9]])
    C1 = np.array([[1.0]])
    Q1 = np.array([[1.0]])
    R1 = np.array([[1.0]])

    # --- Case 2: Multivariate  n_x = n_y = 2 ---
    # A is non-symmetric and stable (eigenvalues 0.8 and 0.5, inside unit circle).
    # C is non-symmetric, so cross-covariances break time-reversal symmetry
    # and S_y can grow linearly with T  (Tong & Zhang 2005).
    A2 = np.array([[0.8, 0.3],
                    [0.0, 0.5]])
    C2 = np.array([[1.0, 0.5],
                    [0.0, 1.0]])
    Q2 = np.eye(2)
    R2 = np.eye(2)

    cases = [
        {"label": r"$n_x = n_y = 1$", "A": A1, "C": C1, "Q": Q1, "R": R1, "tag": "scalar"},
        {"label": r"$n_x = n_y = 2$", "A": A2, "C": C2, "Q": Q2, "R": R2, "tag": "2d"},
    ]

    results = {}

    for case in cases:
        tag = case["tag"]
        A, C, Q, R = case["A"], case["C"], case["Q"], case["R"]
        print(f"\n{'='*60}")
        print(f"  Case: {tag}   A = {A.tolist()},  C = {C.tolist()}")
        print(f"  Q = {Q.tolist()},  R = {R.tolist()}")
        print(f"{'='*60}")

        P, S_mat, K_mat = solve_kalman_steady_state(A, C, Q, R)
        print(f"  Steady-state:  P = {np.round(P, 6).tolist()}")
        print(f"                 S = {np.round(S_mat, 6).tolist()}")
        print(f"                 K = {np.round(K_mat, 6).tolist()}")

        Sy_analytic = np.zeros(len(T_analytic))
        Sy_mc_mean  = np.zeros(len(T_mc))
        Sy_mc_se    = np.zeros(len(T_mc))
        Sy_mc_raw   = np.zeros((N_mc, len(T_mc)))   # all samples for data availability

        for i, T in enumerate(T_analytic):
            Sy_analytic[i] = analytic_entropy_production(A, C, Q, R, int(T))

        for i, T in enumerate(T_mc):
            mc_mean, mc_se, sigmas = mc_entropy_production(A, C, Q, R, int(T), N_mc, rng)
            Sy_mc_mean[i] = mc_mean
            Sy_mc_se[i]   = mc_se
            Sy_mc_raw[:, i] = sigmas
            print(f"  T = {T:3d}:  analytic = {Sy_analytic[T-1]:.6f},  "
                  f"MC = {Sy_mc_mean[i]:.6f} +/- {Sy_mc_se[i]:.6f}")

        results[tag] = {
            "T_analytic": T_analytic,
            "T_mc":       T_mc,
            "analytic":   Sy_analytic,
            "mc_mean":    Sy_mc_mean,
            "mc_se":      Sy_mc_se,
        }

        # --- Save summary data to CSV ---
        csv_file = f"entropy_production_{tag}.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            # Section 0a: Input parameters and simulation settings
            nx = A.shape[0]
            writer.writerow(["# Input parameters"])
            for label, mat in [("A", A), ("C", C), ("Q", Q), ("R", R)]:
                for i in range(mat.shape[0]):
                    row_label = f"{label}({i+1},:)" if mat.shape[0] > 1 else label
                    writer.writerow([row_label] + [f"{mat[i,j]:.10f}" for j in range(mat.shape[1])])
            writer.writerow(["N_mc", N_mc])
            writer.writerow(["seed", seed])
            writer.writerow([])
            # Section 0b: Steady-state Kalman filter quantities
            writer.writerow(["# Steady-state Kalman filter quantities"])
            for label, mat in [("P", P), ("S", S_mat), ("K", K_mat)]:
                for i in range(mat.shape[0]):
                    row_label = f"{label}({i+1},:)" if mat.shape[0] > 1 else label
                    writer.writerow([row_label] + [f"{mat[i,j]:.10f}" for j in range(mat.shape[1])])
            writer.writerow([])
            # Section 1: analytic (all T)
            writer.writerow(["T", "Sy_analytic"])
            for j in range(len(T_analytic)):
                writer.writerow([
                    int(T_analytic[j]),
                    f"{Sy_analytic[j]:.10f}",
                ])
            writer.writerow([])
            # Section 2: Monte Carlo (T = 5, 10, ...)
            writer.writerow(["T", "Sy_mc_mean", "Sy_mc_se"])
            for j in range(len(T_mc)):
                writer.writerow([
                    int(T_mc[j]),
                    f"{Sy_mc_mean[j]:.10f}",
                    f"{Sy_mc_se[j]:.10f}",
                ])
        print(f"  -> Saved {csv_file}")

        # --- Save ALL raw MC samples to CSV (for data availability) ---
        #   Columns: sigma_T5, sigma_T10, ..., sigma_T50
        #   Rows:    sample n = 1, 2, ..., N
        raw_csv = f"entropy_production_{tag}_mc_raw.csv"
        with open(raw_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = [f"sigma_T{int(T)}" for T in T_mc]
            writer.writerow(header)
            for n in range(N_mc):
                writer.writerow([f"{Sy_mc_raw[n, j]:.10f}" for j in range(len(T_mc))])
        print(f"  -> Saved {raw_csv}  ({N_mc} samples x {len(T_mc)} T-values)")

    # =================================================================
    #  Plot: two panels side by side
    # =================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    panel_labels = ["(a)", "(b)"]
    for ax, case, plabel in zip(axes, cases, panel_labels):
        tag = case["tag"]
        r   = results[tag]

        # Analytical curve
        ax.plot(r["T_analytic"], r["analytic"], "k-", linewidth=1.5,
                label="Analytic", zorder=3)

        # Monte Carlo with error bars
        ax.errorbar(r["T_mc"], r["mc_mean"], yerr=r["mc_se"],
                     fmt="o", color="C0", markersize=5, capsize=2,
                     linewidth=0.8, elinewidth=0.8,
                     label=rf"Monte Carlo ($N = {N_mc:,}$)", zorder=2)

        ax.set_xlabel(r"$T$", fontsize=14)
        ax.set_ylabel(r"$\mathcal{S}_y$", fontsize=14)
        ax.set_title(f"{plabel}  {case['label']}", fontsize=20)
        ax.legend(fontsize=14)
        ax.set_xlim(0, 51)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    plot_file = "entropy_production_verification.pdf"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"\n-> Saved {plot_file}")

    plot_png = "entropy_production_verification.png"
    fig.savefig(plot_png, dpi=150, bbox_inches="tight")
    print(f"-> Saved {plot_png}")


if __name__ == "__main__":
    main()
