####################################################################################################
#                                             music.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Differentiable MUSIC layers. The grid variant computes the null spectrum against an     #
#          arbitrary (learned) manifold and extracts peaks with a masked iterative soft-argmax;    #
#          the root variant forms the null polynomial from a modified noise subspace and roots     #
#          it via the companion-matrix eigendecomposition (gridless, differentiable).              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch
import torch.nn as nn


#**************************************************************************************************#
#                                    Class DifferentiableMUSIC                                     #
#**************************************************************************************************#
#                                                                                                  #
# Grid-scan MUSIC: P(theta) = 1 / (|| En^H a(theta) ||^2 + eps) with any manifold A on the grid.   #
# The noise subspace comes from the classical EVD of the sample covariance and is detached by      #
# default (it does not depend on learned parameters); gradients flow through the manifold A.       #
#                                                                                                  #
#**************************************************************************************************#
class DifferentiableMUSIC(nn.Module):
    def __init__(self, grid_size=361, theta_range=(-1.5533, 1.5533), eps=1e-12,
                 peak_window=5, peak_tau=0.05):
        super().__init__()
        self.eps = eps
        self.peak_window = peak_window
        self.peak_tau = peak_tau
        self.register_buffer('grid', torch.linspace(*theta_range, grid_size,
                                                    dtype=torch.float64))

    #***************#
    #   subspaces   #
    #***************#
    def noise_subspace(self, R, n_src, detach=True):
        """En (..., M, M - n_src) from the Hermitian EVD (ascending eigenvalues)."""
        _, evecs = torch.linalg.eigh(R)
        En = evecs[..., :R.shape[-1] - n_src]
        return En.detach() if detach else En

    #**************#
    #   spectrum   #
    #**************#
    def null_spectrum(self, En, A):
        """q(theta) = || En^H a(theta) ||^2 for A (..., M, G) -> (..., G)."""
        return torch.einsum('...mk,...mg->...kg', En.conj(), A.to(En.dtype)).abs().pow(2).sum(-2)

    def spectrum(self, En, A):
        return 1.0 / (self.null_spectrum(En, A) + self.eps)

    #*********************#
    #   peak extraction   #
    #*********************#
    def peaks(self, P, n_src):
        """
        Masked iterative soft-argmax over the grid: per source, take the global argmax of the
        (masked) log-spectrum and refine it with a softmax-weighted mean over a local window
        (differentiable w.r.t. P), then suppress the window. Returns (..., n_src) sorted DoAs.
        """
        logP = torch.log(P.clamp_min(1e-30))
        grid = self.grid.to(P.device)
        G = grid.shape[0]
        offs = torch.arange(-self.peak_window, self.peak_window + 1, device=P.device)
        mask = torch.zeros_like(logP, dtype=torch.bool)
        doas = []
        for _ in range(n_src):
            cur = logP.masked_fill(mask, -torch.inf)
            idx = cur.argmax(dim=-1)                                       # (...,)
            win = (idx[..., None] + offs).clamp(0, G - 1)                  # (..., W)
            vals = torch.gather(logP, -1, win)
            w = torch.softmax(vals / self.peak_tau, dim=-1)
            doas.append((w * grid[win]).sum(-1))
            mask = mask.scatter(-1, win, True)
        return torch.stack(doas, dim=-1).sort(dim=-1).values

    def forward(self, R, A, n_src, detach_subspace=True):
        En = self.noise_subspace(R, n_src, detach=detach_subspace)
        P = self.spectrum(En, A)
        return self.peaks(P, n_src), P


#**************************************************************************************************#
#                                  Class DifferentiableRootMUSIC                                   #
#**************************************************************************************************#
#                                                                                                  #
# Gridless root-MUSIC through a differentiable companion-matrix eigendecomposition. Consumes a     #
# (possibly modified, e.g. D(z)^H En) noise subspace for a uniform linear structure and returns    #
# the n_src DoAs from the roots closest to the unit circle (inside). The polynomial variable is    #
# u = e^(-j*pi*f*s*sin(theta)) for `variable='sin'` (physical ULA of spacing s) or u = e^(j*theta) #
# for `variable='theta'` (Vandermonde wavefield basis of a manifold-separation model).             #
#                                                                                                  #
#**************************************************************************************************#
class DifferentiableRootMUSIC(nn.Module):
    def __init__(self, spacing=1.0, variable='sin', theta_lim=None):
        super().__init__()
        self.spacing = spacing
        self.variable = variable
        self.theta_lim = theta_lim              # 'theta' variable only: keep roots inside the FOV

    #**********************#
    #   polynomial roots   #
    #**********************#
    @staticmethod
    def companion_roots(c):
        """
        Roots of polynomials c (..., n+1) with leading coefficient first, via the eigenvalues of
        the companion matrix (differentiable). Returns (..., n) complex roots.
        """
        n = c.shape[-1] - 1
        lead = c[..., :1]
        lead = torch.where(lead.abs() < 1e-30, torch.full_like(lead, 1e-30), lead)
        monic = c[..., 1:] / lead                                          # (..., n)
        C = torch.zeros(*c.shape[:-1], n, n, dtype=c.dtype, device=c.device)
        C[..., 1:, :-1] = torch.eye(n - 1, dtype=c.dtype, device=c.device)
        C[..., :, -1] = -monic.flip(-1)
        return torch.linalg.eigvals(C)

    def null_polynomial(self, En):
        """Coefficients (..., 2M-1) of q(u) = sum_k tr_k(En En^H) u^k, leading first."""
        Q = En @ En.mH                                                     # (..., M, M)
        M = Q.shape[-1]
        return torch.stack([Q.diagonal(offset=k, dim1=-2, dim2=-1).sum(-1)
                            for k in range(M - 1, -M, -1)], dim=-1)

    #********************#
    #   root selection   #
    #********************#
    def select(self, roots, n_src, f=1.0):
        """n_src angles from candidate roots (..., R): closest to the unit circle (inside
        preferred), FOV-filtered for the 'theta' variable; sorted."""
        mag = roots.abs()
        dist = (1.0 - mag).abs() + torch.where(mag < 1.0, torch.zeros_like(mag),
                                               torch.full_like(mag, 1e3))  # prefer inside roots
        if self.variable == 'theta' and self.theta_lim is not None:        # reject out-of-FOV
            dist = dist + 1e3 * (torch.angle(roots).abs() > self.theta_lim).to(dist.dtype)
        sel = dist.topk(n_src, dim=-1, largest=False).indices
        u = torch.gather(roots, -1, sel)
        if self.variable == 'sin':
            sin_th = (-torch.angle(u) / (torch.pi * f * self.spacing)).clamp(-1.0, 1.0)
            doas = torch.asin(sin_th)
        else:
            doas = torch.angle(u)
        return doas.sort(dim=-1).values

    def forward(self, En, n_src, f=1.0):
        """
        En: (..., M, M - n_src) (modified) noise subspace -> DoAs (..., n_src), sorted.
        """
        return self.select(self.companion_roots(self.null_polynomial(En)), n_src, f=f)
