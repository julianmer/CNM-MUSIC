####################################################################################################
#                                          nearfield.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Near-field (joint angle + distance) estimators: 2D-MUSIC (Huang & Barkat), the          #
#          angle-then-range cascade, and the concentrated maximum-likelihood refiner (MUSIC-       #
#          initialized). All batched-native: X (B, M, T) -> (doas (B, d), ranges (B, d)).          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch

from cnmmusic.estimators.base import Estimator
from cnmmusic.arrays.steering import nearfield_steering_matrix


#**************************************************************************************************#
#                                      Class NearFieldMUSIC                                        #
#**************************************************************************************************#
#                                                                                                  #
# Joint (theta, r) grid scan of the null spectrum with the spherical-wavefront manifold; the d     #
# largest 2-D local maxima give angles and distances simultaneously.                               #
#                                                                                                  #
#**************************************************************************************************#
class NearFieldMUSIC(Estimator):
    joint_axis = 'range'

    def __init__(self, geom=None, config=None, range_grid=(6.0, 250.0, 40)):
        range_grid = (config or {}).get('range_grid', range_grid)   # the scenario's own span
        super().__init__(geom, config)                     # full angle grid, as every scan
        lo, hi, n = range_grid
        self.r_grid = torch.logspace(torch.log10(torch.tensor(lo)),
                                     torch.log10(torch.tensor(hi)), int(n),
                                     dtype=torch.float64, device=self.device)
        G, R = self.grid_size, int(n)
        th = self.grid.repeat(R)                                           # (R*G,)
        rr = self.r_grid.repeat_interleave(G)                              # (R*G,)
        self.A2 = nearfield_steering_matrix(self.geom.positions[None], th[None],
                                            rr[None])[0]                   # (M, R*G)

    def spectrum2d(self, R_hat, n_src, D=None, chunk=4096):
        """
        (B, R, G) joint pseudo-spectrum; optional per-batch manifold modification D (B, M, M).
        Scanned in column chunks: the joint grid matches the CNM scan (grid_size per axis),
        so the full (B, M - d, R * G) product would not fit.
        """
        En = self.noise_subspace(R_hat, n_src)
        A = self.A2 if D is None else D @ self.A2
        A = (A[None] if A.ndim == 2 else A).to(En.dtype)
        B = R_hat.shape[0]
        out = []
        for s in range(0, A.shape[-1], chunk):
            Ac = A[..., s:s + chunk].expand(B, -1, -1)
            out.append(torch.einsum('bmk,bmg->bkg', En.conj(), Ac).abs().pow(2).sum(-2))
        q = torch.cat(out, dim=-1)
        return (1.0 / (q + 1e-12)).reshape(B, len(self.r_grid), self.grid_size)

    def pick_peaks2d(self, P, n_src, excl_deg=4.0):
        """
        Iterative 2-D peak extraction: after each pick the whole ANGULAR column (+- excl_deg,
        all ranges) is suppressed -- the (theta, r) ridge of a close source is elongated in
        range, so plain top-d local maxima put several picks on one source's ridge.
        Returns (doas (B, d), ranges (B, d)), sorted by angle.
        """
        B, R, G = P.shape
        excl = int(excl_deg / torch.rad2deg(self.grid[1] - self.grid[0]).item()) + 1
        work = P.clone().reshape(B, -1)
        doas = torch.empty(B, n_src, dtype=self.grid.dtype, device=self.grid.device)
        rngs = torch.empty(B, n_src, dtype=self.r_grid.dtype, device=self.r_grid.device)
        for k in range(n_src):
            idx = work.argmax(dim=-1)
            g = idx % G
            doas[:, k] = self.grid[g]
            rngs[:, k] = self.r_grid[idx // G]
            lo = (g - excl).clamp(0)
            hi = (g + excl).clamp(max=G - 1)
            ar = torch.arange(G, device=work.device)
            mask = (ar[None, :] >= lo[:, None]) & (ar[None, :] <= hi[:, None])   # (B, G)
            work = work.reshape(B, R, G).masked_fill(mask[:, None, :], 0.0).reshape(B, -1)
        order = doas.argsort(dim=-1)
        return doas.gather(1, order), rngs.gather(1, order)

    def __call__(self, X, n_src=None, D=None):
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        P = self.spectrum2d(self.covariance(X), n_src, D)
        doas, rngs = self.pick_peaks2d(P, n_src)
        return (doas[0], rngs[0]) if single else (doas, rngs)


#**************************************************************************************************#
#                                     Class NearFieldCascade                                       #
#**************************************************************************************************#
#                                                                                                  #
# Angle first (far-field manifold scan: cheap, slightly biased by wavefront curvature), then one   #
# 1-D range scan of the null spectrum at each estimated angle.                                     #
#                                                                                                  #
#**************************************************************************************************#
class NearFieldCascade(Estimator):
    joint_axis = 'range'

    def __init__(self, geom=None, config=None, range_grid=(6.0, 250.0, 64)):
        range_grid = (config or {}).get('range_grid', range_grid)   # the scenario's own span
        super().__init__(geom, config)
        lo, hi, n = range_grid
        self.r_grid = torch.logspace(torch.log10(torch.tensor(lo)),
                                     torch.log10(torch.tensor(hi)), int(n),
                                     dtype=torch.float64, device=self.device)

    def __call__(self, X, n_src=None):
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        R_hat = self.covariance(X)
        En = self.noise_subspace(R_hat, n_src)
        q = torch.einsum('bmk,bmg->bkg', En.conj(),
                         self.A_grid.to(En.dtype)[None]).abs().pow(2).sum(-2)
        doas = self.pick_peaks(1.0 / (q + 1e-12), n_src)                   # far-field angles
        B, d, nR = doas.shape[0], n_src, len(self.r_grid)
        th = doas.repeat_interleave(nR, dim=-1)                            # (B, d*nR)
        rr = self.r_grid.repeat(d)[None].expand(B, -1)
        A = nearfield_steering_matrix(self.geom.positions[None].expand(B, -1, -1), th, rr)
        qr = torch.einsum('bmk,bmg->bkg', En.conj(), A.to(En.dtype)).abs().pow(2).sum(-2)
        rngs = self.r_grid[qr.reshape(B, d, nR).argmin(-1)]                # deepest null over r
        return (doas[0], rngs[0]) if single else (doas, rngs)


#**************************************************************************************************#
#                                       Class NearFieldMLE                                         #
#**************************************************************************************************#
#                                                                                                  #
# Deterministic maximum likelihood: maximize the concentrated likelihood tr(Pi_A(theta, r) R_hat)  #
# over all source angles and distances jointly, initialized at the 2D-MUSIC estimate and refined   #
# with Adam on (theta, log r).                                                                     #
#                                                                                                  #
#**************************************************************************************************#
class NearFieldMLE(Estimator):
    joint_axis = 'range'

    def __init__(self, geom=None, config=None, steps=60, lr=2e-3):
        super().__init__(geom, config)
        self.init = NearFieldMUSIC(geom, config)
        self.steps, self.lr = steps, lr

    def __call__(self, X, n_src=None):
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        R_hat = self.covariance(X)
        doas0, rngs0 = self.init(X, n_src)
        th = doas0.clone().requires_grad_(True)
        logr = rngs0.log().clone().requires_grad_(True)
        pos = self.geom.positions[None].expand(X.shape[0], -1, -1)
        opt = torch.optim.Adam([th, logr], lr=self.lr)
        for _ in range(self.steps):
            A = nearfield_steering_matrix(pos, th, logr.exp())             # (B, M, d)
            gram = A.mH @ A
            eye = torch.eye(n_src, dtype=A.dtype, device=A.device)
            Pi = A @ torch.linalg.solve(gram + 1e-9 * eye, A.mH)
            cost = -torch.einsum('bij,bji->b', Pi, R_hat.to(A.dtype)).real.sum()
            opt.zero_grad()
            cost.backward()
            opt.step()
        doas, rngs = th.detach(), logr.detach().exp()
        order = doas.argsort(dim=-1)
        doas, rngs = doas.gather(1, order), rngs.gather(1, order)
        return (doas[0], rngs[0]) if single else (doas, rngs)


#**************************************************************************************************#
#                                      Class NearFieldMVDR                                         #
#**************************************************************************************************#
#                                                                                                  #
# The joint (theta, r) scan under the Capon back-end: P = 1 / (a^H R^-1 a) on the same 2-D grid    #
# with the same 2-D peak extraction, so it differs from NearFieldMUSIC only in the spectrum.       #
#                                                                                                  #
#**************************************************************************************************#
class NearFieldMVDR(NearFieldMUSIC):
    def spectrum2d(self, R_hat, n_src, D=None, chunk=4096):
        """(B, R, G) joint Capon spectrum, scanned in column chunks like the MUSIC variant."""
        B, M = R_hat.shape[0], R_hat.shape[-1]
        load = 1e-6 * R_hat.diagonal(dim1=-2, dim2=-1).real.mean(-1)[..., None, None]
        Rinv = torch.linalg.inv(R_hat + load * torch.eye(M, dtype=R_hat.dtype,
                                                         device=R_hat.device))
        A = self.A2 if D is None else D @ self.A2
        A = (A[None] if A.ndim == 2 else A).to(Rinv.dtype)
        out = []
        for s in range(0, A.shape[-1], chunk):
            Ac = A[..., s:s + chunk].expand(B, -1, -1)
            out.append(torch.einsum('bmg,bmn,bng->bg', Ac.conj(), Rinv, Ac).real)
        q = torch.cat(out, dim=-1)
        return (1.0 / (q + 1e-12)).reshape(B, len(self.r_grid), self.grid_size)


#**************************************************************************************************#
#                                  Class NearFieldCascadeMVDR                                      #
#**************************************************************************************************#
#                                                                                                  #
# The two-stage scan under the Capon back-end: far-field Capon angles first, then the range that   #
# maximizes the Capon spectrum along each estimated angle.                                         #
#                                                                                                  #
#**************************************************************************************************#
class NearFieldCascadeMVDR(NearFieldCascade):
    @staticmethod
    def _inv(R_hat):
        M = R_hat.shape[-1]
        load = 1e-6 * R_hat.diagonal(dim1=-2, dim2=-1).real.mean(-1)[..., None, None]
        return torch.linalg.inv(R_hat + load * torch.eye(M, dtype=R_hat.dtype,
                                                         device=R_hat.device))

    def __call__(self, X, n_src=None):
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        R_hat = self.covariance(X)
        Rinv = self._inv(R_hat)
        A = self.A_grid.to(Rinv.dtype)
        q = torch.einsum('mg,bmn,ng->bg', A.conj(), Rinv, A).real
        doas = self.pick_peaks(1.0 / (q + 1e-12), n_src)                   # far-field angles
        B, d, nR = doas.shape[0], n_src, len(self.r_grid)
        th = doas.repeat_interleave(nR, dim=-1)                            # (B, d*nR)
        rr = self.r_grid.repeat(d)[None].expand(B, -1)
        Ar = nearfield_steering_matrix(self.geom.positions[None].expand(B, -1, -1), th, rr)
        Ar = Ar.to(Rinv.dtype)
        qr = torch.einsum('bmg,bmn,bng->bg', Ar.conj(), Rinv, Ar).real
        rngs = self.r_grid[qr.reshape(B, d, nR).argmin(-1)]                # Capon peak over r
        return (doas[0], rngs[0]) if single else (doas, rngs)
