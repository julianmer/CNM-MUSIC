####################################################################################################
#                                             zzb.py                                               #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Ziv-Zakai bound (ZZB) for DoA estimation (Bell, Steinberg, Ephraim & Van Trees          #
#          form): relates the MSE to the minimum error probability of binary hypothesis tests      #
#          between shifted DoA hypotheses, integrated over the uniform DoA prior with valley       #
#          filling. Unlike the CRB it accounts for large ambiguity errors, so it stays             #
#          informative through the threshold region and approaches the prior variance at zero      #
#          SNR. Per-source genie form (the other sources held at their true DoAs -- a valid,       #
#          slightly optimistic bound); the test error probability uses the Gaussian (CLT)          #
#          approximation of the log-likelihood ratio over T snapshots.                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import math
import torch

from cnmmusic.arrays.steering import steering_matrix


#*********************************#
#   two-point error probability   #
#*********************************#
def _pe_gauss(R_a, R_b, Rinv_a, Rinv_b, logdet_a, logdet_b, T):
    """
    Minimum error probability (equal priors) between CN(0, R_a) and CN(0, R_b) from T iid
    snapshots, via the normal approximation of the log-likelihood ratio. All inputs batched.
    """
    B0 = torch.eye(R_a.shape[-1], dtype=R_a.dtype, device=R_a.device) - Rinv_b @ R_a
    B1 = Rinv_a @ R_b - torch.eye(R_a.shape[-1], dtype=R_a.dtype, device=R_a.device)
    c = (logdet_a - logdet_b).real
    tr = lambda X: X.diagonal(dim1=-2, dim2=-1).sum(-1).real
    mu0 = c + tr(B0)                                               # E[LLR] under H_a (<= 0)
    mu1 = c + tr(B1)                                               # E[LLR] under H_b (>= 0)
    s0 = tr(B0 @ B0).clamp_min(1e-30).sqrt()
    s1 = tr(B1 @ B1).clamp_min(1e-30).sqrt()
    q = lambda x: 0.5 * torch.erfc(x / math.sqrt(2.0))
    rootT = math.sqrt(T)
    return 0.5 * (q(-rootT * mu0 / s0) + q(rootT * mu1 / s1))


#**********************#
#   zzb from a batch   #
#**********************#
def zzb_from_batch(batch, positions, theta_range, n_scenes=32, n_phi=48):
    """
    Ziv-Zakai bound on the DoA RMSPE [rad] for a simulator batch: uniform prior over
    theta_range, per-source genie tests, valley filling, averaged over n_scenes scenes and
    all sources. Returns a scalar [rad] comparable to the CRB-RMSPE.
    """
    doas = batch['doas'][:, 0]
    B, d = doas.shape
    take = torch.linspace(0, B - 1, min(n_scenes, B)).long()
    T = batch['X'].shape[-1]
    lo, hi = theta_range
    W = hi - lo
    grid = torch.linspace(lo, hi, n_phi, dtype=torch.float64)
    dstep = W / (n_phi - 1)

    # the Pe transition sits at h ~ 1/sqrt(T * FIM), far below the coarse phi spacing at large
    # T -- a log-spaced fine-h segment below dstep keeps the small-error region resolved
    h_fine = torch.logspace(math.log10(1e-5), math.log10(dstep), 24, dtype=torch.float64)[:-1]
    phi_sub = grid[::3]                                            # fine-h test locations

    mses = []
    for i in take.tolist():
        A = steering_matrix(positions[None], doas[i][None])[0]     # (M, d) true manifold
        a_grid = steering_matrix(positions[None], grid[None])[0]   # (M, G)
        R_s = batch['R_s'][i, 0].to(A.dtype)
        s2 = batch['sigma2'][i].to(A.dtype)
        M = A.shape[0]
        eye = torch.eye(M, dtype=A.dtype)

        def cov(phis, j):
            Aj = A[None].repeat(len(phis), 1, 1)
            Aj[:, :, j] = steering_matrix(positions[None], phis[None])[0].T
            return Aj @ R_s @ Aj.mH + s2 * eye

        for j in range(d):
            Rg = cov(grid, j)
            Rinv = torch.linalg.inv(Rg)
            logdet = torch.linalg.slogdet(Rg).logabsdet

            Ah = torch.zeros(n_phi, dtype=torch.float64)           # coarse: h = k * dstep
            for k in range(1, n_phi):
                a, b = torch.arange(n_phi - k), torch.arange(k, n_phi)
                pe = _pe_gauss(Rg[a], Rg[b], Rinv[a], Rinv[b], logdet[a], logdet[b], T)
                Ah[k] = (2.0 / W) * (W - k * dstep) * pe.mean()
            R0 = cov(phi_sub, j)                                   # fine h below the grid step
            Ri0, ld0 = torch.linalg.inv(R0), torch.linalg.slogdet(R0).logabsdet
            Ah_f = torch.zeros(len(h_fine), dtype=torch.float64)
            for k, hf in enumerate(h_fine.tolist()):
                R1 = cov(phi_sub + hf, j)
                pe = _pe_gauss(R0, R1, Ri0, torch.linalg.inv(R1), ld0,
                               torch.linalg.slogdet(R1).logabsdet, T)
                Ah_f[k] = (2.0 / W) * (W - hf) * pe.mean()

            h = torch.cat([torch.zeros(1, dtype=torch.float64), h_fine,
                           torch.arange(1, n_phi, dtype=torch.float64) * dstep])
            Ah = torch.cat([torch.ones(1, dtype=torch.float64), Ah_f, Ah[1:]])
            Ah = Ah.flip(0).cummax(0).values.flip(0)               # valley filling
            mses.append(0.5 * torch.trapezoid(h * Ah, h))
    return torch.stack(mses).mean().sqrt()
