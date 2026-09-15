####################################################################################################
#                                             crb.py                                               #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Stochastic (unconditional) Cramer-Rao bound for DoA estimation with the TRUE            #
#          (imperfection-perturbed) manifold — the genie bound all methods are measured against.   #
#          The Stoica-Nehorai form uses the source covariance R_s directly (never its inverse),    #
#          so it remains valid for coherent sources (rank-deficient R_s), where it correctly       #
#          yields a larger bound than the independent-source case.                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import math

import torch

from cnmmusic.arrays.steering import steering_matrix, steering_derivative


#********************#
#   stochastic crb   #
#********************#
def stochastic_fim(A, dA, R_s, sigma2, T, Q=None):
    """
    Batched stochastic Fisher information (Stoica & Nehorai): A, dA (B, M, d) true manifold and
    its theta-derivative at the true DoAs; R_s (B, d, d); sigma2 (B,); T snapshots.
    Returns the DoA Fisher information matrices (B, d, d).

    Q (B, M, M) is a KNOWN spatially coloured noise covariance (noise = sigma^2 Q). Whitening
    A and dA by its Cholesky factor makes this exact form valid for coloured noise: with
    L L^H = Q, L^-1 R L^-H = (L^-1 A) R_s (L^-1 A)^H + sigma^2 I. Q = None is white noise.
    """
    if Q is not None:
        L = torch.linalg.cholesky(Q.to(A.dtype))
        A = torch.linalg.solve_triangular(L, A, upper=False)
        dA = torch.linalg.solve_triangular(L, dA, upper=False)
    B, M, d = A.shape
    eye = torch.eye(M, dtype=A.dtype, device=A.device)
    Rx = A @ R_s.to(A.dtype) @ A.mH + sigma2[..., None, None].to(A.dtype) * eye
    Rx_inv = torch.linalg.inv(Rx)

    AhA_inv = torch.linalg.inv(A.mH @ A)
    proj = eye - A @ AhA_inv @ A.mH                                # orthogonal projector
    H = (dA.mH @ proj @ dA) * (R_s.to(A.dtype) @ A.mH @ Rx_inv @ A @ R_s.to(A.dtype)).transpose(-1, -2)
    return (2.0 * T / sigma2[..., None, None]) * H.real


def stochastic_crb(A, dA, R_s, sigma2, T):
    """Per-scene stochastic CRB: per-source DoA variances (B, d) [rad^2]."""
    fim = stochastic_fim(A, dA, R_s, sigma2, T)
    return torch.linalg.inv(fim).diagonal(dim1=-2, dim2=-1)


#**********************************#
#   bound for the mean statistic   #
#**********************************#
def mean_statistic_factor(d):
    """
    The reported metric is the MEAN over scenes of the per-scene RMSPE
    e = sqrt((1/d) sum_i err_i^2), but the CRB bounds the error STANDARD DEVIATION. For an
    efficient unbiased estimator with d Gaussian errors of std sigma, e/sigma is
    sqrt(chi^2_d / d), so

        E[e] = c_d * sigma ,   c_d = sqrt(2/d) * Gamma((d+1)/2) / Gamma(d/2)  <= 1

    (0.798 at d=1, 0.886 at d=2, 0.921 at d=3, -> 1 as d grows). Scaling the per-scene bound
    by c_d makes it the correct floor for the mean statistic: an efficient estimator then
    touches it at 1.0 instead of sitting 8-20% below it.
    """
    return math.sqrt(2.0 / d) * math.exp(math.lgamma((d + 1) / 2) - math.lgamma(d / 2))


#**********************#
#   crb from a batch   #
#**********************#
def crb_from_batch(batch, positions, mode='scene', curved=False, delta=1e-4,
                   prior_var=None, reduce=True):
    """
    Rebuilds the true perturbed manifold and its theta-derivative from the batch ground truths
    (flattened imperfections: [delta_pos 3M | gain M | phase M | coupling Re M^2 | Im M^2])
    and returns the CRB-RMSPE [rad] over the batch (segment 0).

    Defaults reproduce the classic protocol: plane-wave manifold, mean of per-scene bounds.
    Opt-ins (OFF by default): curved=True evaluates the Fisher information at the true curved
    wavefront when the batch carries finite ranges; mode='bayesian' (Van Trees) averages the
    Fisher information over the batch and inverts once -- the bound on the ensemble-mean error,
    immune to the heavy tail of near-degenerate draws; prior_var [rad^2] clips each per-scene
    variance at the uniform-prior bound (W^2/12) -- beyond it the local bound is vacuous.
    """
    doas = batch['doas'][:, 0]
    B, d = doas.shape
    M = positions.shape[0]
    imp = batch['imperfect']
    dpos = imp[:, :3 * M].reshape(B, M, 3)
    gain = imp[:, 3 * M:4 * M]
    phase = imp[:, 4 * M:5 * M]
    C = torch.complex(imp[:, 5 * M:5 * M + M * M].reshape(B, M, M),
                      imp[:, 5 * M + M * M:].reshape(B, M, M))

    pos = positions[None] + dpos                                   # (B, M, 3)
    D = C @ torch.diag_embed(torch.polar(gain, phase).to(torch.complex128))
    rngs = batch.get('ranges') if curved else None
    rngs = rngs[:, 0] if (rngs is not None and rngs.ndim == 3) else rngs
    if rngs is not None and torch.isfinite(rngs).all():            # true curved wavefront
        from cnmmusic.arrays.steering import nearfield_steering_matrix
        A = D @ nearfield_steering_matrix(pos, doas, rngs)
        dA = D @ (nearfield_steering_matrix(pos, doas + delta, rngs)
                  - nearfield_steering_matrix(pos, doas - delta, rngs)) / (2 * delta)
    else:                                                          # plane waves
        A = D @ steering_matrix(pos, doas)
        dA = D @ steering_derivative(pos, doas)
    Q = batch.get('Q')                       # spatially coloured noise: sigma^2 Q, Q known
    if Q is not None and Q.ndim == 4:
        Q = Q[:, 0]
    fim = stochastic_fim(A, dA, batch['R_s'][:, 0], batch['sigma2'], batch['X'].shape[-1], Q=Q)
    if mode == 'bayesian':
        var = torch.linalg.inv(fim.mean(0)).diagonal(dim1=-2, dim2=-1)
        return var.clamp_min(0.0).mean().sqrt()
    var = torch.linalg.inv(fim).diagonal(dim1=-2, dim2=-1).clamp_min(0.0)
    if prior_var is not None:
        var = var.clamp(max=prior_var)
    per_scene = var.mean(-1).sqrt() * mean_statistic_factor(d)     # (B,) rad, bound on E[e]
    return per_scene.pow(2).mean().sqrt() if reduce else per_scene


#**************************#
#   joint (theta, r) crb   #
#**************************#
def _joint_var(A, dA_th, dA_aux, R_s, sigma2, T):
    """Stochastic CRB for (theta, aux) with 2 parameters per source: (var_theta, var_aux)."""
    d = dA_th.shape[-1]
    M = A.shape[-2]
    eye = torch.eye(M, dtype=A.dtype, device=A.device)
    Rx = A @ R_s.to(A.dtype) @ A.mH + sigma2[..., None, None].to(A.dtype) * eye
    Rx_inv = torch.linalg.inv(Rx)
    proj = eye - A @ torch.linalg.inv(A.mH @ A) @ A.mH
    G = (R_s.to(A.dtype) @ A.mH @ Rx_inv @ A @ R_s.to(A.dtype)).transpose(-1, -2)

    D = torch.cat([dA_th, dA_aux], dim=-1)                             # (B, M, 2d)
    H = (D.mH @ proj @ D) * G.repeat(1, 2, 2)
    fim = (2.0 * T / sigma2[..., None, None]) * H.real
    var = torch.linalg.inv(fim).diagonal(dim1=-2, dim2=-1).clamp_min(0.0)
    return var[..., :d], var[..., d:]


def nearfield_crb(pos, doas, rngs, R_s, sigma2, T, delta=1e-4):
    """
    Stochastic CRB for joint angle + distance estimation (2 parameters per source), with
    central finite-difference manifold derivatives. pos (M, 3); doas, rngs (B, d);
    returns (var_theta (B, d), var_r (B, d)).
    """
    from cnmmusic.arrays.steering import nearfield_steering_matrix
    B, d = doas.shape
    P = pos[None].expand(B, -1, -1)
    A = nearfield_steering_matrix(P, doas, rngs)
    dA_th = (nearfield_steering_matrix(P, doas + delta, rngs)
             - nearfield_steering_matrix(P, doas - delta, rngs)) / (2 * delta)
    dA_r = (nearfield_steering_matrix(P, doas, rngs + delta)
            - nearfield_steering_matrix(P, doas, rngs - delta)) / (2 * delta)
    return _joint_var(A, dA_th, dA_r, R_s, sigma2, T)


def freq_crb(pos, doas, freqs, R_s, sigma2, T, delta=1e-4):
    """
    Stochastic CRB for joint angle + carrier-frequency estimation. Uses a per-source frequency
    parametrization (each source gets its own f), which is conservative for the shared-carrier
    case. pos (M, 3); doas (B, d); freqs (B,) normalized to the carrier;
    returns (var_theta (B, d), var_f (B, d)).
    """
    B, d = doas.shape
    P = pos[None].expand(B, -1, -1)
    # freqs is (B,) when one carrier is shared by every source in a scene, or (B, d) when
    # each source carries its own (the broadband-tones case)
    freqs = torch.as_tensor(freqs, dtype=doas.dtype, device=doas.device)
    F = freqs[..., :d] if freqs.ndim == 2 else freqs[:, None].expand(B, d)
    A = steering_matrix(P, doas, f=F)
    dA_th = (steering_matrix(P, doas + delta, f=F)
             - steering_matrix(P, doas - delta, f=F)) / (2 * delta)
    dA_f = (steering_matrix(P, doas, f=F + delta)
            - steering_matrix(P, doas, f=F - delta)) / (2 * delta)
    return _joint_var(A, dA_th, dA_f, R_s, sigma2, T)


#***************************#
#   hybrid (bayesian) crb   #
#***************************#
def _fim_general(A, Bs, R_s, sigma2, T):
    """
    Stochastic FIM over an arbitrary parameter set, each entering through the manifold as
    B_k = dA/d(alpha_k) (B, M, d). Generalizes stochastic_fim (which is the special case
    B_i = d_i e_i^T) by Schur-complementing R_s and sigma^2 out of the Slepian-Bangs FIM:

        [J]_kl = (2T / sigma^2) * Re tr( R_s A^H R^-1 A R_s  B_l^H Pi_A_perp B_k )

    Stoica & Nehorai, IEEE TASSP 38(10):1783-1795, 1990 (DOI 10.1109/29.60109); the
    multi-parameter-per-source Kronecker form is eq. (71) of Khamidullina, Podkurkov &
    Haardt, IEEE TSP 69:3220-3234, 2021 (DOI 10.1109/TSP.2021.3082469).
    """
    B_, M, d = A.shape
    eye = torch.eye(M, dtype=A.dtype, device=A.device)
    R_s = R_s.to(A.dtype)
    Rx = A @ R_s @ A.mH + sigma2[..., None, None].to(A.dtype) * eye
    proj = eye - A @ torch.linalg.inv(A.mH @ A) @ A.mH
    H = R_s @ A.mH @ torch.linalg.inv(Rx) @ A @ R_s                    # (B, d, d)
    P = len(Bs)
    G = torch.stack([proj @ b for b in Bs], dim=1)                     # (B, P, M, d)
    J = torch.empty(B_, P, P, dtype=torch.float64, device=A.device)
    for k in range(P):
        # tr(H Bl^H Pi Bk) for every l at once
        Mkl = torch.einsum('bij,bpjk->bpik', H, G.mH @ G[:, k][:, None])
        J[:, :, k] = Mkl.diagonal(dim1=-2, dim2=-1).sum(-1).real
    return (2.0 * T / sigma2[..., None, None]) * J


def calibration_jacobians(positions, doas, gain, phase, C):
    """
    dA/d(calibration parameter) for the gain / phase / x-position of each sensor, with the
    perturbed manifold A = C diag(g e^{j phi}) A0(x, theta), A0[m,k] = exp(-j pi x_m sin th_k).
    Writing c_m for column m of C, alpha_m for row m of A0 and w_m = g_m e^{j phi_m}:

        dA/dg_m   = c_m (w_m / g_m) alpha_m
        dA/dphi_m = c_m (j w_m)     alpha_m
        dA/ddelta_m = c_m w_m (-j pi sin theta) * alpha_m

    Sensor 0 is the gauge reference and is excluded, so 3(M-1) matrices come back, each
    (B, M, d). NOTE: a ULA needs TWO phase/position references for full identifiability
    (Rockah & Schultheiss, IEEE TASSP 35(3):286-299, 1987) -- this single-reference gauge is
    only sufficient because the hybrid prior regularizes the remaining null directions.
    """
    B_, M, _ = positions.shape
    A0 = steering_matrix(positions, doas)                              # (B, M, d)
    w = (gain * torch.polar(torch.ones_like(phase), phase)).to(A0.dtype)      # (B, M)
    g = gain.to(A0.dtype)
    ramp = (-1j * torch.pi * torch.sin(doas)).to(A0.dtype)             # (B, d)
    out = []
    for m in range(1, M):                                              # gauge: skip sensor 0
        c_m, row, w_m = C[:, :, m].to(A0.dtype), A0[:, m, :], w[:, m:m + 1]
        for scale in (row * (w_m / g[:, m:m + 1]),                     # gain
                      row * (1j * w_m),                                # phase
                      row * w_m * ramp):                               # x-position
            out.append(c_m[:, :, None] * scale[:, None, :])
    return out


def hybrid_crb_from_batch(batch, positions, rho, max_gain=0.2, max_pos=0.2,
                          max_phase_deg=30.0, prior_var=None, reduce=True):
    """
    Hybrid (Bayesian) CRB: the DoAs are deterministic unknowns while the calibration errors
    are RANDOM nuisances with the generator's own prior, eta_k ~ U(-rho*max_k, rho*max_k),
    so var(eta_k) = (rho*max_k)^2 / 3. The prior Fisher information is added to the joint
    FIM's calibration block:

        J_hybrid = J_data(theta, eta) + diag(0, ..., 0, 1/var(eta_1), ..., 1/var(eta_p))
        CRB(theta) = [J_hybrid^-1]_{1:d, 1:d}

    This is the right bound for a nominally linear array, where the array CANNOT be
    self-calibrated at all (Rockah & Schultheiss, IEEE TASSP 35(3):286-299, 1987, DOI
    10.1109/TASSP.1987.1165144) so the pure self-calibration FIM is singular and its
    constrained inverse is gauge-dependent. The prior regularizes the ambiguous directions,
    giving a unique, gauge-free bound that equals the known-array (genie) bound at rho = 0
    and grows monotonically with rho. Hybrid bounds for array auto-calibration: Rockah &
    Schultheiss 1987 (above) and Viberg & Swindlehurst, IEEE TSP 42(12):3495-3507, 1994
    (DOI 10.1109/78.340783).

    Mutual coupling is treated as KNOWN: the generator makes it deterministic given rho
    (gamma = rho * gamma_0), so it carries no prior spread to add.
    """
    doas = batch['doas'][:, 0]
    B_, d = doas.shape
    M = positions.shape[0]
    imp = batch['imperfect']
    dpos = imp[:, :3 * M].reshape(B_, M, 3)
    gain = imp[:, 3 * M:4 * M]
    phase = imp[:, 4 * M:5 * M]
    C = torch.complex(imp[:, 5 * M:5 * M + M * M].reshape(B_, M, M),
                      imp[:, 5 * M + M * M:].reshape(B_, M, M))
    pos = positions[None] + dpos
    D = C @ torch.diag_embed(torch.polar(gain, phase).to(torch.complex128))
    A = D @ steering_matrix(pos, doas)
    dA = D @ steering_derivative(pos, doas)

    theta_B = [torch.zeros_like(A) for _ in range(d)]                  # B_i = d_i e_i^T
    for i in range(d):
        theta_B[i][:, :, i] = dA[:, :, i]
    calib_B = calibration_jacobians(pos, doas, gain, phase, C)
    J = _fim_general(A, theta_B + calib_B, batch['R_s'][:, 0], batch['sigma2'],
                     batch['X'].shape[-1])

    # prior FIM on the calibration block. rho = 0 means the array is exactly nominal, i.e.
    # the calibration parameters are known -- infinite prior information, so the bound is
    # the known-array one; without this the unregularized FIM is singular (a ULA cannot be
    # self-calibrated) and inflates the bound by orders of magnitude.
    if rho <= 0:
        return crb_from_batch(batch, positions, prior_var=prior_var, reduce=reduce)
    var = torch.tensor([(rho * max_gain) ** 2, (rho * math.radians(max_phase_deg)) ** 2,
                        (rho * max_pos) ** 2], dtype=torch.float64) / 3.0
    prior = var.repeat(M - 1).reciprocal()
    J[:, d:, d:] += torch.diag_embed(prior.expand(B_, -1))
    var_theta = torch.linalg.inv(J)[:, :d, :d].diagonal(dim1=-2, dim2=-1).clamp_min(0.0)
    if prior_var is not None:
        var_theta = var_theta.clamp(max=prior_var)
    per_scene = var_theta.mean(-1).sqrt() * mean_statistic_factor(d)
    return per_scene.pow(2).mean().sqrt() if reduce else per_scene
