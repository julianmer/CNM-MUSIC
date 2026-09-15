####################################################################################################
#                                           classical.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Classical DoA baselines (MUSIC, Root-MUSIC, ESPRIT, MVDR, spatial smoothing, oracle),   #
#          all batched-native: (B, M, T) snapshots in, (B, d) DoAs out, on CPU or GPU.             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch

from cnmmusic.estimators.base import Estimator
from cnmmusic.archs.music import DifferentiableRootMUSIC
from cnmmusic.arrays.steering import steering_matrix


#**************************************************************************************************#
#                                            Class MUSIC                                           #
#**************************************************************************************************#
#                                                                                                  #
# The canonical noise-subspace grid scan: P = 1 / || En^H a(theta) ||^2, one einsum per batch.     #
# An explicit manifold A (M, G) or (B, M, G) may be passed (hook for learned/oracle manifolds).    #
#                                                                                                  #
#**************************************************************************************************#
class MUSIC(Estimator):
    def spectrum(self, R, n_src, A=None):
        A = self.A_grid if A is None else A
        A = A[None] if A.ndim == 2 else A
        En = self.noise_subspace(R, n_src)
        q = torch.einsum('bmk,bmg->bkg', En.conj(), A.to(En.dtype)).abs().pow(2).sum(-2)
        return 1.0 / (q + 1e-12)

    def __call__(self, X, n_src=None, A=None):
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        doas = self.pick_peaks(self.spectrum(self.covariance(X), n_src, A), n_src)
        return doas[0] if single else doas


#**************************************************************************************************#
#                                        Class OracleMUSIC                                         #
#**************************************************************************************************#
#                                                                                                  #
# MUSIC scanned against the ground-truth perturbed manifold (upper performance bound).             #
#                                                                                                  #
#**************************************************************************************************#
class OracleMUSIC(MUSIC):
    def __call__(self, X, n_src=None, A_true=None):
        assert A_true is not None, 'OracleMUSIC needs the ground-truth manifold on its grid'
        return super().__call__(X, n_src, A=A_true)


#**************************************************************************************************#
#                                         Class RootMUSIC                                          #
#**************************************************************************************************#
#                                                                                                  #
# Gridless polynomial rooting for half-wavelength ULAs via the batched companion-matrix            #
# eigendecomposition (thousands of polynomials rooted in one call).                                #
#                                                                                                  #
#**************************************************************************************************#
class RootMUSIC(Estimator):
    def __init__(self, geom=None, config=None):
        super().__init__(geom, config)
        self.rooter = DifferentiableRootMUSIC()

    def __call__(self, X, n_src=None):
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        En = self.noise_subspace(self.covariance(X), n_src)
        with torch.no_grad():
            doas = self.rooter(En, n_src)
        return doas[0] if single else doas


#**************************************************************************************************#
#                                           Class ESPRIT                                           #
#**************************************************************************************************#
#                                                                                                  #
# Shift-invariance on the two maximal overlapping ULA subarrays (batched least squares).           #
#                                                                                                  #
#**************************************************************************************************#
class ESPRIT(Estimator):
    def __call__(self, X, n_src=None):
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        _, evecs = torch.linalg.eigh(self.covariance(X))
        Es = evecs[..., -n_src:]                                   # signal subspace (B, M, d)
        Phi = torch.linalg.lstsq(Es[..., :-1, :], Es[..., 1:, :]).solution
        eig = torch.linalg.eigvals(Phi)
        sin_th = (-torch.angle(eig) / torch.pi).clamp(-1.0, 1.0)
        doas = torch.asin(sin_th).sort(dim=-1).values
        return doas[0] if single else doas


#**************************************************************************************************#
#                                          Class FreqMUSIC                                         #
#**************************************************************************************************#
#                                                                                                  #
# Joint (theta, f) MUSIC for off-carrier narrowband signals: the MUSIC spectrum is scanned over    #
# a carrier-frequency grid; f_hat picks the sharpest spectrum per scene (the null depth            #
# collapses off the true carrier), DoAs are read at f_hat. Returns (doas (B, d), f_hat (B,)).      #
#                                                                                                  #
#**************************************************************************************************#
class FreqMUSIC(MUSIC):
    joint_axis = 'freq'

    def __init__(self, geom=None, config=None, f_grid=(0.75, 1.25, 41)):
        f_grid = (config or {}).get('f_grid', f_grid)               # the scenario's own span
        super().__init__(geom, config)
        lo, hi, n = f_grid
        self.f_grid = torch.linspace(lo, hi, n, dtype=torch.float64, device=self.device)
        self.A_freq = torch.stack(                                     # (F, M, G)
            [steering_matrix(self.geom.positions[None], self.grid[None], f=float(f))[0]
             for f in self.f_grid])

    def pick_peaks2d(self, P, n_src, excl_deg=4.0):
        """
        Iterative 2-D peak extraction over the (f, theta) map: after each pick the whole
        ANGULAR column (+- excl_deg, all carriers) is suppressed, so the ridge of one source
        cannot absorb several picks. Returns (doas (B, d), freqs (B, d)), sorted by angle.
        """
        B, F, G = P.shape
        excl = int(excl_deg / torch.rad2deg(self.grid[1] - self.grid[0]).item()) + 1
        work = P.clone().reshape(B, -1)
        doas = torch.empty(B, n_src, dtype=self.grid.dtype, device=self.grid.device)
        freqs = torch.empty(B, n_src, dtype=self.f_grid.dtype, device=self.f_grid.device)
        for k in range(n_src):
            idx = work.argmax(dim=-1)
            g = idx % G
            doas[:, k] = self.grid[g]
            freqs[:, k] = self.f_grid[idx // G]
            lo, hi = (g - excl).clamp(0), (g + excl).clamp(max=G - 1)
            ar = torch.arange(G, device=work.device)
            mask = (ar[None, :] >= lo[:, None]) & (ar[None, :] <= hi[:, None])   # (B, G)
            work = work.reshape(B, F, G).masked_fill(mask[:, None, :], 0.0).reshape(B, -1)
        order = doas.argsort(dim=-1)
        return doas.gather(1, order), freqs.gather(1, order)

    def __call__(self, X, n_src=None):
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        En = self.noise_subspace(self.covariance(X), n_src)
        P = self.map2d(self.covariance(X), En)                     # (B, F, G)
        doas, f_hat = self.pick_peaks2d(P, n_src)
        return (doas[0], f_hat[0]) if single else (doas, f_hat)

    def map2d(self, R_hat, En):
        """(B, F, G) joint null spectrum over the (f, theta) grid."""
        return torch.stack([1.0 / (torch.einsum('bmk,mg->bkg', En.conj(), Af.to(En.dtype))
                                   .abs().pow(2).sum(-2) + 1e-12)
                            for Af in self.A_freq], dim=1)

    def spectrum_map(self, X, n_src=None):
        """(B, F, G) pseudo-spectrum over the (f, theta) grid (single input -> (F, G))."""
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        En = self.noise_subspace(self.covariance(X), n_src)
        P = torch.stack([1.0 / (torch.einsum('bmk,mg->bkg', En.conj(), Af.to(En.dtype))
                                .abs().pow(2).sum(-2) + 1e-12)
                         for Af in self.A_freq], dim=1)
        return P[0] if single else P


#**************************************************************************************************#
#                                            Class MVDR                                            #
#**************************************************************************************************#
#                                                                                                  #
# Capon spectrum P = 1 / (a^H R^-1 a) with diagonal loading, batched solve + einsum.               #
#                                                                                                  #
#**************************************************************************************************#
class MVDR(Estimator):
    def spectrum(self, R, n_src=None):
        """Capon spectrum 1 / (a^H R^-1 a) over the grid, batched: R (B, M, M) -> (B, G)."""
        M = R.shape[-1]
        load = 1e-6 * R.diagonal(dim1=-2, dim2=-1).real.mean(-1)[..., None, None]
        eye = torch.eye(M, dtype=R.dtype, device=R.device)
        Rinv = torch.linalg.inv(R + load * eye)
        A = self.A_grid.to(R.dtype)
        return 1.0 / (torch.einsum('mg,bmn,ng->bg', A.conj(), Rinv, A).real + 1e-12)

    def __call__(self, X, n_src=None):
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        R = self.covariance(X)
        M = R.shape[-1]
        load = 1e-6 * R.diagonal(dim1=-2, dim2=-1).real.mean(-1)[..., None, None]
        eye = torch.eye(M, dtype=R.dtype, device=R.device)
        Rinv = torch.linalg.inv(R + load * eye)
        A = self.A_grid.to(R.dtype)
        q = torch.einsum('mg,bmn,ng->bg', A.conj(), Rinv, A).real
        doas = self.pick_peaks(1.0 / (q + 1e-12), n_src)
        return doas[0] if single else doas


#**************************************************************************************************#
#                                   Class SpatialSmoothingMUSIC                                    #
#**************************************************************************************************#
#                                                                                                  #
# Forward-backward spatial smoothing over ULA subarrays (coherent sources), then the MUSIC scan.   #
# Subarray covariances are formed with one unfold instead of a Python loop.                        #
#                                                                                                  #
#**************************************************************************************************#
class SpatialSmoothingMUSIC(MUSIC):
    spectrum_on_snapshots = True                   # overlays must hand us snapshots, not R_full

    def __init__(self, geom=None, config=None, subarray=None):
        super().__init__(geom, config)
        self.subarray = subarray if subarray is not None else self.geom.M - 2
        self.A_grid = steering_matrix(self.geom.positions[:self.subarray][None],
                                      self.grid[None])[0]

    def smooth(self, X):
        """Forward-backward spatially smoothed covariance (..., L, L) from snapshots."""
        L, T = self.subarray, X.shape[-1]
        sub = X.unfold(-2, L, 1).transpose(-1, -2)                 # (..., n_sub, L, T)
        R = (sub @ sub.mH).mean(-3) / T                            # (..., L, L)
        J = torch.flip(torch.eye(L, dtype=R.dtype, device=R.device), dims=[0])
        return 0.5 * (R + J @ R.conj() @ J)                        # forward-backward

    def spectrum(self, R_or_X, n_src, A=None):
        """Accepts snapshots (which are smoothed first) or an already-smoothed covariance."""
        Z = torch.as_tensor(R_or_X)
        single = Z.ndim == 2
        if not (Z.shape[-1] == Z.shape[-2] == self.subarray):
            Z = self.smooth(Z[None] if single else Z)
        elif single:
            Z = Z[None]
        P = super().spectrum(Z, n_src, A)
        return P[0] if single else P

    def __call__(self, X, n_src=None):
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        doas = self.pick_peaks(self.spectrum(self.smooth(X), n_src), n_src)
        return doas[0] if single else doas


#**************************************************************************************************#
#                                     Class OracleRootMUSIC                                        #
#**************************************************************************************************#
#                                                                                                  #
# Gridless rooting with the TRUE gain/phase/coupling modification applied to the noise subspace    #
# (En_tilde = D_true^H En): the upper bound for manifold-corrected rooting. Exact when position    #
# errors are absent; with position errors the manifold is not exactly D * a0(theta) and the bound  #
# becomes approximate.                                                                             #
#                                                                                                  #
#**************************************************************************************************#
class OracleRootMUSIC(RootMUSIC):
    def __call__(self, X, n_src=None, imperfect=None):
        assert imperfect is not None, 'OracleRootMUSIC needs the flattened imperfection params'
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        M = X.shape[-2]
        gain = imperfect[..., 3 * M:4 * M]
        phase = imperfect[..., 4 * M:5 * M]
        c_re = imperfect[..., 5 * M:5 * M + M * M].reshape(-1, M, M)
        c_im = imperfect[..., 5 * M + M * M:].reshape(-1, M, M)
        D = torch.complex(c_re, c_im) @ torch.diag_embed(
            torch.polar(gain, phase).to(torch.complex128))
        En = self.noise_subspace(self.covariance(X), n_src)
        with torch.no_grad():
            doas = self.rooter(D.mH @ En, n_src)
        return doas[0] if single else doas


#**************************************************************************************************#
#                                             Class MLE                                            #
#**************************************************************************************************#
#                                                                                                  #
# Deterministic maximum likelihood via Alternating Projection (Ziskind & Wax, 1988): SELF-         #
# initializing (sources placed greedily by global 1-D scans of the concentrated likelihood),       #
# then cyclic global 1-D re-optimization of each angle. No external initializer, each scan is      #
# global, and no full-rank source covariance is required (works for coherent sources).             #
#                                                                                                  #
#**************************************************************************************************#
class MLE(Estimator):
    def __init__(self, geom=None, config=None, sweeps=3):
        super().__init__(geom, config)
        self.sweeps = sweeps

    def _gain(self, R, A_fix):
        """
        Likelihood gain of adding one candidate angle over the whole grid, given the fixed
        sources A_fix (B, M, k): g(theta) = a^H P R P a / a^H P a with P = I - Pi_{A_fix}.
        Returns (B, G).
        """
        B, M = R.shape[0], R.shape[-1]
        eye = torch.eye(M, dtype=R.dtype, device=R.device)
        if A_fix is None:
            P = eye.expand(B, M, M)
        else:
            k = A_fix.shape[-1]
            gram = A_fix.mH @ A_fix + 1e-9 * torch.eye(k, dtype=R.dtype, device=R.device)
            P = eye - A_fix @ torch.linalg.solve(gram, A_fix.mH)
        A = self.A_grid.to(R.dtype)                                # (M, G)
        C1 = P @ R @ P
        num = torch.einsum('mg,bmn,ng->bg', A.conj(), C1, A).real
        den = torch.einsum('mg,bmn,ng->bg', A.conj(), P, A).real.clamp_min(1e-12)
        return num / den

    def __call__(self, X, n_src=None):
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        R = self.covariance(X)
        B = R.shape[0]
        A0 = self.A_grid.to(R.dtype)

        # greedy self-initialization: place sources one by one via global 1-D scans
        idx = torch.empty(B, n_src, dtype=torch.long)
        A_fix = None
        for k in range(n_src):
            idx[:, k] = self._gain(R, A_fix).argmax(-1)
            cols = A0[:, idx[:, :k + 1]].permute(1, 0, 2)          # (B, M, k+1)
            A_fix = cols

        # cyclic refinement: global 1-D re-scan of each angle with the others fixed
        for _ in range(self.sweeps):
            for k in range(n_src):
                others = [j for j in range(n_src) if j != k]
                A_fix = A0[:, idx[:, others]].permute(1, 0, 2) if others else None
                idx[:, k] = self._gain(R, A_fix).argmax(-1)

        doas = self.grid[idx].sort(dim=-1).values
        return doas[0] if single else doas


#**************************************************************************************************#
#                                        Class FreqMVDR                                            #
#**************************************************************************************************#
#                                                                                                  #
# The joint (theta, f) scan under the Capon back-end: the same 2-D grid and peak extraction as     #
# FreqMUSIC, with P = 1 / (a^H R^-1 a) in place of the null spectrum.                              #
#                                                                                                  #
#**************************************************************************************************#
class FreqMVDR(FreqMUSIC):
    def map2d(self, R_hat, En):
        """(B, F, G) joint Capon spectrum over the (f, theta) grid."""
        B, M = R_hat.shape[0], R_hat.shape[-1]
        load = 1e-6 * R_hat.diagonal(dim1=-2, dim2=-1).real.mean(-1)[..., None, None]
        Rinv = torch.linalg.inv(R_hat + load * torch.eye(M, dtype=R_hat.dtype,
                                                         device=R_hat.device))
        out = []
        for Af in self.A_freq:
            A = Af.to(Rinv.dtype)
            q = torch.einsum('mg,bmn,ng->bg', A.conj(), Rinv, A).real
            out.append(1.0 / (q + 1e-12))
        return torch.stack(out, dim=1)


#**************************************************************************************************#
#                                       Class FreqCascade                                          #
#**************************************************************************************************#
#                                                                                                  #
# The two-stage counterpart of NearFieldCascade on the carrier axis: angles first from the scan    #
# at the nominal carrier (cheap, biased when the true carrier is far from f_c), then one 1-D       #
# carrier scan of the null spectrum at each estimated angle.                                       #
#                                                                                                  #
#**************************************************************************************************#
class FreqCascade(FreqMUSIC):
    def _angles(self, R_hat, En, n_src):
        """Stage one: the ordinary far-field scan at the nominal carrier."""
        q = torch.einsum('bmk,bmg->bkg', En.conj(),
                         self.A_grid.to(En.dtype)[None]).abs().pow(2).sum(-2)
        return self.pick_peaks(1.0 / (q + 1e-12), n_src)

    def _carrier_cost(self, R_hat, En, A):
        """Stage two cost along the carrier grid: the null spectrum denominator."""
        return torch.einsum('bmk,bmg->bkg', En.conj(), A.to(En.dtype)).abs().pow(2).sum(-2)

    def __call__(self, X, n_src=None):
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        R_hat = self.covariance(X)
        En = self.noise_subspace(R_hat, n_src)
        doas = self._angles(R_hat, En, n_src)
        B, d, nF = doas.shape[0], n_src, len(self.f_grid)
        th = doas.repeat_interleave(nF, dim=-1)                        # (B, d*nF)
        ff = self.f_grid.repeat(d)[None].expand(B, -1)
        A = steering_matrix(self.geom.positions[None].expand(B, -1, -1), th, f=ff)
        qf = self._carrier_cost(R_hat, En, A)
        freqs = self.f_grid[qf.reshape(B, d, nF).argmin(-1)]           # deepest null over f
        return (doas[0], freqs[0]) if single else (doas, freqs)


#**************************************************************************************************#
#                                     Class FreqCascadeMVDR                                        #
#**************************************************************************************************#
#                                                                                                  #
# The carrier cascade under the Capon back-end: Capon angles at the nominal carrier, then the      #
# carrier maximizing the Capon spectrum along each estimated angle.                                #
#                                                                                                  #
#**************************************************************************************************#
class FreqCascadeMVDR(FreqCascade):
    @staticmethod
    def _inv(R_hat):
        M = R_hat.shape[-1]
        load = 1e-6 * R_hat.diagonal(dim1=-2, dim2=-1).real.mean(-1)[..., None, None]
        return torch.linalg.inv(R_hat + load * torch.eye(M, dtype=R_hat.dtype,
                                                         device=R_hat.device))

    def _angles(self, R_hat, En, n_src):
        Rinv = self._inv(R_hat)
        A = self.A_grid.to(Rinv.dtype)
        q = torch.einsum('mg,bmn,ng->bg', A.conj(), Rinv, A).real
        return self.pick_peaks(1.0 / (q + 1e-12), n_src)

    def _carrier_cost(self, R_hat, En, A):
        Rinv = self._inv(R_hat)
        A = A.to(Rinv.dtype)
        return torch.einsum('bmg,bmn,bng->bg', A.conj(), Rinv, A).real

#**************************************************************************************************#
#                                       Class FreqESPRIT                                           #
#**************************************************************************************************#
#                                                                                                  #
# Joint angle-frequency estimation (JAFE) by space-time ESPRIT: L time-lagged copies of the        #
# snapshots are stacked into one space-time vector, whose signal subspace carries two shift        #
# invariances -- a temporal shift by one sample (eigenvalue e^(j pi f)) and a sensor shift by one  #
# element (eigenvalue e^(-j pi f sin theta)). Both are read from the SAME eigenvectors, so the     #
# carrier and the angle come out paired, closed form and gridless.                                 #
#                                                                                                  #
#**************************************************************************************************#
class FreqESPRIT(Estimator):
    joint_axis = 'freq'

    def __init__(self, geom=None, config=None, lags=4):
        super().__init__(geom, config)
        self.lags = int((config or {}).get('jafe_lags', lags))

    @torch.no_grad()
    def __call__(self, X, n_src=None):
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        B, M, T = X.shape
        L = max(2, min(self.lags, T - n_src))                  # need T - L + 1 snapshots left
        Tp = T - L + 1
        Z = torch.cat([X[..., l:l + Tp] for l in range(L)], dim=-2)        # (B, M*L, Tp)
        Rz = Z @ Z.mH / Tp
        evals, evecs = torch.linalg.eigh(Rz)
        Us = evecs[..., -n_src:]                                           # (B, M*L, d)

        # temporal invariance: lag blocks 0..L-2 against 1..L-1  ->  e^(j pi f)
        Psi_t = torch.linalg.pinv(Us[:, :M * (L - 1)]) @ Us[:, M:]
        mu, Tm = torch.linalg.eig(Psi_t)
        f_hat = torch.angle(mu).real.double() / torch.pi

        # spatial invariance: sensors 0..M-2 against 1..M-1 inside every lag block, rotated by
        # the SAME eigenvectors, so each angle stays paired with its carrier
        i1 = [l * M + m for l in range(L) for m in range(M - 1)]
        i2 = [l * M + m + 1 for l in range(L) for m in range(M - 1)]
        Psi_s = torch.linalg.pinv(Us[:, i1]) @ Us[:, i2]
        Ds = torch.linalg.solve(Tm, Psi_s.to(Tm.dtype) @ Tm)
        nu = torch.diagonal(Ds, dim1=-2, dim2=-1)
        f_safe = f_hat.abs().clamp_min(0.05)
        sin_th = (-torch.angle(nu).real.double() / (torch.pi * f_safe)).clamp(-1.0, 1.0)
        doas = torch.asin(sin_th)

        order = doas.argsort(dim=-1)
        doas, f_hat = doas.gather(1, order), f_hat.gather(1, order)
        return (doas[0], f_hat[0]) if single else (doas, f_hat)
