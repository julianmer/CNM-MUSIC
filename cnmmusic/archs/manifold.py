####################################################################################################
#                                           manifold.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/07/26                                                                                #
#                                                                                                  #
# Purpose: The learned manifold-separation generator:                                              #
#              a(theta, r, f | z) = G(z) . ( d_theta(theta) (x) d_r(1/r) (x) d_f(f) )              #
#          Each axis has a Vandermonde basis in its own unit-circle variable, so every axis is     #
#          rootable: theta via u = e^(j*theta), distance via w = e^(j*s_r*(1/r)), frequency via    #
#          v = e^(j*s_f*(f-1)). The coefficient tensor G(z) = G0 + dG(z) is predicted from the     #
#          observation latent z (zero-init head), with G0 the least-squares fit of the nominal     #
#          manifold, so the model starts exactly at physics. Pinned axes collapse to order 1.      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch
import torch.nn as nn

from cnmmusic.arrays.steering import steering_matrix, nearfield_steering_matrix, gauge_fix
from cnmmusic.archs.layers import zero_linear


#**************************************************************************************************#
#                                     Class SeparatedManifold                                      #
#**************************************************************************************************#
class SeparatedManifold(nn.Module):
    def __init__(self, M=8, z_dim=16, L_theta=32, L_r=1, L_f=1,
                 u_max=0.2, f_halfspan=0.2, ridge=1e-8):
        super().__init__()
        self.M, self.L_theta, self.L_r, self.L_f = M, L_theta, L_r, L_f
        self.u_max, self.f_halfspan, self.ridge = u_max, f_halfspan, ridge
        # axis variables map onto a quarter/half circle (monotone, unambiguous decode):
        # u = 1/r in [0, u_max] -> phase [0, pi/2]; f - 1 in [-h, h] -> phase [-pi/2, pi/2]
        self.s_r = (torch.pi / 2) / u_max if L_r > 1 else 0.0
        self.s_f = (torch.pi / 2) / f_halfspan if L_f > 1 else 0.0
        L = L_theta * L_r * L_f
        self.head = zero_linear(z_dim, 2 * M * L)
        self.register_buffer('G0', torch.zeros(M, L_theta, L_r, L_f, dtype=torch.complex128))
        self.register_buffer('fitted', torch.tensor(False))

    #****************#
    #   axis bases   #
    #****************#
    def d_theta(self, theta):
        """(..., L_theta) with centered orders (matches the rooting variable u = e^(j*theta))."""
        l = torch.arange(self.L_theta, dtype=theta.dtype, device=theta.device) \
            - (self.L_theta - 1) / 2
        return torch.exp(1j * l * theta[..., None])

    def d_r(self, u):
        """(..., L_r) in w = e^(j*s_r*u), orders 0..L_r-1; u = 1/r (0 = far field)."""
        m = torch.arange(self.L_r, dtype=u.dtype, device=u.device)
        return torch.exp(1j * self.s_r * m * u[..., None])

    def d_f(self, f):
        """(..., L_f) in v = e^(j*s_f*(f-1)), orders 0..L_f-1."""
        n = torch.arange(self.L_f, dtype=f.dtype, device=f.device)
        return torch.exp(1j * self.s_f * n * (f[..., None] - 1.0))

    #************************#
    #   nominal fit (once)   #
    #************************#
    @torch.no_grad()
    def fit_nominal(self, positions):
        """
        Least-squares G0 of the nominal manifold over a dense (theta, u, f) product grid.
        The theta grid spans the FULL circle: the harmonics are orthogonal there (perfect
        conditioning) and the fit is globally valid, so rooting sees no unconstrained arc.
        Positions are centered (pure per-theta phase shift, invisible to subspace tests) to
        keep the needed harmonic order at ~pi * aperture/2 << L_theta.
        Called once per geometry (single-geometry training; refit when the geometry changes).
        """
        dev = positions.device
        positions = positions - positions.mean(dim=0, keepdim=True)
        th = torch.linspace(-torch.pi, torch.pi, 1441, dtype=torch.float64, device=dev)[:-1]
        us = (torch.linspace(0.0, self.u_max, 4 * self.L_r + 1, dtype=torch.float64, device=dev)
              if self.L_r > 1 else torch.zeros(1, dtype=torch.float64, device=dev))
        fs = (torch.linspace(1.0 - self.f_halfspan, 1.0 + self.f_halfspan, 4 * self.L_f + 1,
                             dtype=torch.float64, device=dev)
              if self.L_f > 1 else torch.ones(1, dtype=torch.float64, device=dev))
        cols_A, cols_D = [], []
        for u in us:
            for f in fs:
                if u > 0:
                    r = torch.full_like(th, float(1.0 / u))
                    A = nearfield_steering_matrix(positions[None], th[None], r[None],
                                                  f=float(f))[0]
                else:
                    A = steering_matrix(positions[None], th[None], f=float(f))[0]
                cols_A.append(A)                                            # (M, Nth)
                d = (self.d_theta(th)[:, :, None, None]
                     * self.d_r(u.expand(th.shape))[:, None, :, None]
                     * self.d_f(f.expand(th.shape))[:, None, None, :])
                cols_D.append(d.reshape(th.shape[0], -1).T)                 # (L, Nth)
        A0 = torch.cat(cols_A, dim=-1)                                      # (M, N)
        D = torch.cat(cols_D, dim=-1)                                       # (L, N)
        # SVD pseudo-inverse: harmonics restricted to the FOV arc are near-degenerate, so a
        # plain normal-equations solve blows up -- pinv truncates the null directions instead
        G0 = A0 @ torch.linalg.pinv(D)
        self.G0.copy_(G0.reshape(self.M, self.L_theta, self.L_r, self.L_f))
        self.fitted.fill_(True)

    #************************#
    #   coefficient tensor   #
    #************************#
    def G(self, z):
        """(B, M, L_theta, L_r, L_f): nominal fit plus the zero-init learned correction."""
        assert bool(self.fitted), 'call fit_nominal(positions) before using the manifold'
        dG = self.head(z).reshape(-1, self.M, self.L_theta, self.L_r, self.L_f, 2)
        return self.G0 + torch.complex(dG[..., 0], dG[..., 1]).to(self.G0.dtype)

    #********************************************************#
    #   effective per-axis sampling matrices (for rooting)   #
    #********************************************************#
    def theta_matrix(self, z, u=None, f=None):
        """(B, M, L_theta): contract the r and f axes at given values (anchors by default)."""
        G = self.G(z)
        B = G.shape[0]
        u = torch.zeros(B, dtype=torch.float64, device=G.device) if u is None \
            else torch.as_tensor(u, dtype=torch.float64, device=G.device).expand(B)
        f = torch.ones(B, dtype=torch.float64, device=G.device) if f is None \
            else torch.as_tensor(f, dtype=torch.float64, device=G.device).expand(B)
        G = torch.einsum('bmtrf,br->bmtf', G, self.d_r(u))
        return torch.einsum('bmtf,bf->bmt', G, self.d_f(f))

    def r_matrix(self, z, theta, f=None):
        """(B, d, M, L_r): contract theta (per source, e.g. the true DoAs) and f."""
        G = self.G(z)
        B = G.shape[0]
        f = torch.ones(B, dtype=torch.float64, device=G.device) if f is None \
            else torch.as_tensor(f, dtype=torch.float64, device=G.device).expand(B)
        G = torch.einsum('bmtrf,bf->bmtr', G, self.d_f(f))
        return torch.einsum('bmtr,bdt->bdmr', G, self.d_theta(theta))

    def f_matrix(self, z, theta, u=None):
        """(B, d, M, L_f): contract theta (per source) and the range axis."""
        G = self.G(z)
        B = G.shape[0]
        if u is None:
            u = torch.zeros(B, theta.shape[-1], dtype=torch.float64, device=G.device)
        G = torch.einsum('bmtrf,bdr->bdmtf', G, self.d_r(u))
        return torch.einsum('bdmtf,bdt->bdmf', G, self.d_theta(theta))

    #****************************************#
    #   manifold evaluation (plots, tests)   #
    #****************************************#
    def forward(self, theta, z, positions=None, r=None, f=None):
        """
        a_hat (B, M, Q) at query points theta (Q,) or (B, Q); r, f optional per-point values.
        Evaluation utility for visualization and conformance checks -- estimation is rooting.
        """
        G = self.G(z)
        B = G.shape[0]
        theta = torch.atleast_2d(torch.as_tensor(theta, dtype=torch.float64,
                                                 device=G.device)).expand(B, -1)
        Q = theta.shape[-1]
        if r is None:
            u = torch.zeros_like(theta)
        else:
            r = torch.atleast_2d(torch.as_tensor(r, dtype=torch.float64,
                                                 device=G.device)).expand(B, Q)
            u = torch.where(torch.isfinite(r), 1.0 / r.clamp_min(1e-9), torch.zeros_like(r))
        f = torch.ones_like(theta) if f is None \
            else torch.atleast_2d(torch.as_tensor(f, dtype=torch.float64,
                                                  device=G.device)).expand(B, Q)
        d = (self.d_theta(theta)[..., :, None, None]
             * self.d_r(u)[..., None, :, None]
             * self.d_f(f)[..., None, None, :]).reshape(B, Q, -1)          # (B, Q, L)
        A = torch.einsum('bml,bql->bmq', G.reshape(B, self.M, -1), d)
        return gauge_fix(A)
