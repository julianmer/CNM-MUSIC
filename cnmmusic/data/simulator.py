####################################################################################################
#                                           simulator.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Batched narrowband array-signal simulator emitting the full training sample contract.   #
#          Wide-domain sampling per condition/batch: geometry mixture with per-geometry field of   #
#          view and aperture range, SNR / snapshot (log) / source-count / correlation / power-     #
#          imbalance ranges, near-field mixing, and bounds-based imperfection draws. SNR           #
#          convention: unit-power complex noise, source power 10^(SNR/10).                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import math
import torch

from cnmmusic.arrays.geometry import ArrayGeometry
from cnmmusic.arrays.imperfections import ImperfectionModel
from cnmmusic.arrays.steering import gauge_fix, nearfield_steering_matrix


#**************************************************************************************************#
#                                     Class NarrowbandSimulator                                    #
#**************************************************************************************************#
#                                                                                                  #
# One 'condition' = one imperfection/scene draw shared by K snapshot segments with independent     #
# DOAs (structural z anti-leakage). EVERY ranged value is drawn per batch element -- T, d, SNR,    #
# frequency, imperfections, correlation, power imbalance, distances; only the geometry kind is     #
# per batch (one positions tensor). 'snapshots' and 'n_src' are (B,) tensors; X is zero-padded     #
# to the batch T_max and doas NaN-padded to d_max (padding is storage only: R_hat divides by       #
# each element's own T, and consumers slice (T, d)-groups before any computation).                 #
#                                                                                                  #
#**************************************************************************************************#
class NarrowbandSimulator:
    def __init__(self, config=None):
        self.default_config = {
            # geometry (new-style mixture; falls back to fixed 'geometry')
            'geometry': 'ula',            # ArrayGeometry instance or factory name (legacy path)
            'geometries': None,           # list of kinds -> per-batch draw (overrides 'geometry')
            'freeze_geometry': False,     # draw the geometry ONCE and reuse it every batch
                                          # (per-aperture training on random arrays)
            'coarray': False,             # sparse-array front-end: lag-average R, Hankel the
                                          # windows into pseudo-snapshots of the virtual ULA
            'spacing': None,              # (lo, hi) adjacent-element spacing, per-batch draw
            'fov_deg': None,              # {kind: +-half-FOV in degrees}
            'M': 8,
            # imperfections
            'imperfections': None,        # ImperfectionModel (bounds or rho mode) or None
            'rho_range': None,            # legacy per-condition rho draw (rho-mode models only)
            # estimation difficulty
            'n_src_range': (2, 2),
            'snr_range': (0.0, 0.0),
            'snapshots_range': (100, 100),
            'log_snapshots': False,       # draw T log-uniformly over snapshots_range
            'power_imbalance_db': 0.0,    # legacy scalar spread
            'power_imbalance_range': None,# (lo, hi) per-condition spread
            'min_sep': 0.07,              # minimum angular separation [rad]
            'max_sep': None,              # optional maximum adjacent separation [rad]
            'fixed_sep': None,            # exact adjacent separation [rad]; overrides min/max_sep
            'theta_range': (-1.5533, 1.5533),  # legacy fixed FOV [rad]
            # correlation (1.0 = fully coherent: rank-1 source covariance)
            'source_corr': 0.0,           # legacy scalar
            'source_corr_range': None,    # (lo, hi) per-condition draw
            # spatially colored noise: per-condition AR-1 coefficient of the Toeplitz noise
            # covariance Q_ij = nc^|i-j| (unit diagonal keeps the SNR convention); None = white
            'noise_corr_range': None,
            # distributed sources: per-source angular spread [deg], coherent rays -> one
            # effective steering vector per source; None = point sources
            'spread_deg': None,
            # sensor failures: (lo, hi) integer count of dead elements per scene, drawn
            # uniformly, at random positions among sensors 1..M-1 (the gauge reference
            # sensor 0 stays alive); a dead element outputs noise only
            'n_failed': None,
            # moving sources: per-source total drift over the window [deg], linear, truth at
            # the window center; None = static
            'motion_deg': None,
            # source distance: None = far field; (lo, hi) = per-source log-uniform draw
            'range_range': None,
            # band
            'freq_range': (1.0, 1.0),     # source frequency f/fc per condition
            'per_source_freq': False,     # tones: every source draws its OWN carrier (B, d_max)
            # temporal signal (DA-MUSIC broadband suites):
            # gauss = white envelopes | tone = s_d(t) = s_bar_d exp(pi j f_d n) | ofdm =
            # per-source subcarrier comb over [carrier, carrier + bandwidth), per-bin steering
            'signal': 'gauss',
            'subcarriers': 0,
            'bandwidth': 0.0,
            # bookkeeping
            'segments': 1,
            'grid_size': 721,
            'seed': 42,
        }
        if config is not None:
            self.default_config.update({k: v for k, v in config.items()
                                        if k in self.default_config})
        self.__dict__.update(self.default_config)

        self.geom = (self.geometry if isinstance(self.geometry, ArrayGeometry)
                     else getattr(ArrayGeometry, self.geometry)(self.M))
        self.M = self.geom.M
        self.imperfections = (self.imperfections if self.imperfections is not None
                              else ImperfectionModel(kinds=(), rho=0.0))
        self.rng = torch.Generator().manual_seed(self.seed)
        self.grid = torch.linspace(*self.theta_range, self.grid_size, dtype=torch.float64)
        self._frozen = self._draw_geometry() if self.freeze_geometry else None
        self._ca_pos = None                     # coarray lag maps, cached per geometry
        if (self.spread_deg is not None or self.motion_deg is not None) and \
                (self.range_range is not None or self.per_source_freq
                 or (self.spread_deg is not None and self.motion_deg is not None)):
            raise ValueError('spread/motion are far-field single-carrier axes (one at a time)')
        if self.signal not in ('gauss', 'tone', 'ofdm'):
            raise ValueError(f'unknown signal model: {self.signal}')
        corr_hi = (max(self.source_corr_range) if self.source_corr_range is not None
                   else self.source_corr)
        if self.signal != 'gauss' and (self.range_range is not None or corr_hi > 0
                                       or self.spread_deg is not None
                                       or self.motion_deg is not None):
            raise ValueError('tone/ofdm signals are far-field uncorrelated-source axes')

    #***********************#
    #   batch-level draws   #
    #***********************#
    def _uniform(self, lo, hi):
        return lo + (hi - lo) * torch.rand(1, generator=self.rng, dtype=torch.float64).item()

    def _draw_geometry(self):
        """Geometry kind + element spacing per batch; returns (geom, theta_range)."""
        if self.geometries is None:
            return self.geom, self.theta_range
        kind = self.geometries[torch.randint(len(self.geometries), (1,),
                                             generator=self.rng).item()]
        sp = self._uniform(*self.spacing) if self.spacing is not None else 1.0
        if kind == 'ula':
            geom = ArrayGeometry.ula(self.M, spacing=sp)
        elif kind == 'uca':                               # adjacent-chord spacing -> radius
            geom = ArrayGeometry.uca(self.M, radius=sp / (2 * math.sin(math.pi / self.M)))
        elif kind == 'ura':
            mx = max(2, int(self.M ** 0.5))
            geom = ArrayGeometry.ura(mx, max(self.M // mx, 1), spacing=sp)
        elif kind == 'nested':                            # coarray-contiguous sparse array
            geom = ArrayGeometry.nested(self.M, spacing=sp)
        elif kind == 'nula':                              # irregular gaps, mean spacing = sp
            gaps = 0.5 + torch.rand(self.M - 1, generator=self.rng, dtype=torch.float64)
            p = torch.cat([torch.zeros(1, dtype=torch.float64), gaps.cumsum(0)])
            geom = ArrayGeometry.nonuniform_la(p * sp * (self.M - 1) / p[-1])
        elif kind == 'random_planar':                     # overall extent = sp * (M - 1)
            geom = ArrayGeometry.random_planar(self.M, sp * (self.M - 1), rng=self.rng)
        else:
            raise ValueError(f'unknown geometry kind: {kind}')
        fov = (self.fov_deg.get(kind) if isinstance(self.fov_deg, dict)
               else self.fov_deg) if self.fov_deg is not None else None
        if fov is not None:                          # scalar = one FOV for every geometry
            half = math.radians(float(fov)) - 1e-3
            return geom, (-half, half)
        return geom, self.theta_range

    def _draw_snapshots(self):
        lo, hi = self.snapshots_range
        if not self.log_snapshots or lo == hi:
            return torch.randint(lo, hi + 1, (1,), generator=self.rng).item()
        u = self._uniform(math.log(lo), math.log(hi))
        return max(lo, min(hi, int(round(math.exp(u)))))

    #***************************#
    #   condition-level draws   #
    #***************************#
    def _sample_doas(self, d, theta_range):
        lo, hi = theta_range
        if self.fixed_sep is not None and d > 1:
            span = (d - 1) * self.fixed_sep
            if span >= hi - lo:
                raise RuntimeError('fixed_sep span exceeds the field of view')
            c = self._uniform(lo + span / 2, hi - span / 2)
            return c - span / 2 + self.fixed_sep * torch.arange(d, dtype=torch.float64)
        for _ in range(1000):
            doas = lo + (hi - lo) * torch.rand(d, generator=self.rng, dtype=torch.float64)
            doas = doas.sort().values
            ok = d == 1 or (doas.diff() >= self.min_sep).all()
            if ok and self.max_sep is not None and d > 1:
                ok = (doas.diff() <= self.max_sep).all()
            if ok:
                return doas
        if self.max_sep is None:            # dense packings (large d * min_sep): exact
            # construction -- uniform draws in the gap-reduced interval, sorted, plus the
            # accumulated minimum gaps; the correct conditional-uniform distribution
            u = (lo + (hi - lo - (d - 1) * self.min_sep)
                 * torch.rand(d, generator=self.rng, dtype=torch.float64)).sort().values
            return u + self.min_sep * torch.arange(d, dtype=torch.float64)
        raise RuntimeError('could not sample DOAs with the requested separation')

    def _spread_steering(self, geom, params, th, f, spreads):
        """Coherently distributed sources: one EFFECTIVE steering vector per source,
        a_eff = sum_l g_l a(theta + delta_l) over n_rays coherent rays inside the spread
        (unit expected power). th (d,), spreads (d,) [rad] -> (M, d)."""
        n_rays = 16
        cols = []
        for i in range(th.shape[0]):
            delta = (torch.rand(n_rays, generator=self.rng,
                                dtype=torch.float64) - 0.5) * spreads[i]
            g = torch.randn(n_rays, 2, generator=self.rng,
                            dtype=torch.float64) / (2 * n_rays) ** 0.5
            a = self.imperfections.true_manifold(geom, params, th[i] + delta, f=f,
                                                 gauge=False)
            cols.append(a.to(torch.complex128) @ torch.complex(g[:, 0], g[:, 1]))
        return torch.stack(cols, dim=-1)

    def _motion_steering(self, geom, params, th, f, drifts, T):
        """Per-snapshot steering under linear drift centered on the reported truth.
        th (d,), drifts (d,) [rad over the window] -> (M, T, d)."""
        tf = torch.linspace(-0.5, 0.5, T, dtype=torch.float64)
        th_t = th[None, :] + drifts[None, :] * tf[:, None]                  # (T, d)
        A = self.imperfections.true_manifold(geom, params, th_t.reshape(-1), f=f,
                                             gauge=False)
        return A.reshape(self.M, T, th.shape[0])

    def _broadband_steering(self, geom, params, th, f):
        """Imperfect far-field steering at per-tuple (theta, f): th (n,), f (n,) -> (M, n)."""
        pos = (geom.positions + params.delta_pos)[None]
        rr = torch.full((1, th.shape[0]), torch.inf, dtype=torch.float64)
        a0 = nearfield_steering_matrix(pos, th[None], rr, f[None])[0]
        dgp = (params.gain * torch.exp(1j * params.phase)).to(a0.dtype)
        return params.coupling.to(a0.dtype) @ (dgp[:, None] * a0)

    def _source_cov(self, d, corr):
        spread = (self._uniform(*self.power_imbalance_range)
                  if self.power_imbalance_range is not None else self.power_imbalance_db)
        p_db = torch.tensor([self._uniform(-spread / 2, spread / 2) for _ in range(d)],
                            dtype=torch.float64)
        powers = 10.0 ** (p_db / 10.0)
        C = torch.full((d, d), float(corr), dtype=torch.float64).fill_diagonal_(1.0)
        P = powers.sqrt().diag()
        return (P @ C @ P).to(torch.complex128)                        # corr = 1 -> rank-1

    #********************#
    #   batch sampling   #
    #********************#
    def sample(self, batch_size):
        """
        Returns a dict batch. Shapes: X (B, K, M, T_max), doas/ranges (B, K, d_max),
        a_true_grid (B, M, G), R_hat / R_true (B, K, M, M) (the sample covariance and the
        exact ensemble covariance A R_s A^H + I); grid (G,) spans the batch FOV.
        T and d are per-element draws: X is zero-padded to T_max (padded snapshots contribute
        nothing to R_hat, which divides by each element's own T), doas NaN-padded to d_max,
        and 'snapshots' / 'n_src' are (B,) tensors.
        """
        geom, theta_range = self._frozen if self._frozen is not None else self._draw_geometry()
        grid = torch.linspace(*theta_range, self.grid_size, dtype=torch.float64)
        K, M, G = self.segments, self.M, self.grid_size
        d_cap = self._coarray_maps(geom)[2] if self.coarray else self.M - 1
        d_lo, d_hi = self.n_src_range[0], min(self.n_src_range[1], d_cap)
        ds = torch.randint(d_lo, d_hi + 1, (batch_size,), generator=self.rng)
        Ts = torch.tensor([self._draw_snapshots() for _ in range(batch_size)])
        d_max, T_max = int(ds.max()), int(Ts.max())

        X = torch.zeros(batch_size, K, M, T_max, dtype=torch.complex128)
        doas = torch.full((batch_size, K, d_max), torch.nan, dtype=torch.float64)
        ranges = torch.full((batch_size, K, d_max), torch.inf, dtype=torch.float64)
        drift = torch.zeros(batch_size, d_max, dtype=torch.float64)
        spread = torch.zeros(batch_size, d_max, dtype=torch.float64)
        Q_out = torch.eye(M, dtype=torch.complex128).expand(batch_size, M, M).clone()
        a_true_grid = torch.empty(batch_size, M, G, dtype=torch.complex128)
        a_true_doa = torch.zeros(batch_size, K, M, d_max, dtype=torch.complex128)
        R_s = torch.zeros(batch_size, K, d_max, d_max, dtype=torch.complex128)
        R_true = torch.empty(batch_size, K, M, M, dtype=torch.complex128)
        snr = torch.empty(batch_size, dtype=torch.float64)
        freqs = (torch.full((batch_size, d_max), torch.nan, dtype=torch.float64)
                 if self.per_source_freq else torch.ones(batch_size, dtype=torch.float64))
        coherent = torch.zeros(batch_size, dtype=torch.bool)
        nearfield = torch.zeros(batch_size, dtype=torch.bool)
        imperfect = []

        eye = torch.eye(M, dtype=torch.complex128)
        for b in range(batch_size):
            d, T = int(ds[b]), int(Ts[b])
            rho_b = (self._uniform(*self.rho_range) if self.rho_range is not None else None)
            params = self.imperfections.sample(M, rho=rho_b)
            if self.noise_corr_range is not None:
                nc = self._uniform(*self.noise_corr_range)
                k = torch.arange(M, dtype=torch.float64)
                Q = (nc ** (k[:, None] - k[None, :]).abs()).to(torch.complex128)
                Lq = torch.linalg.cholesky(Q + 1e-9 * eye)
            else:
                Q, Lq = eye, None
            spread_b = (torch.tensor([math.radians(self._uniform(*self.spread_deg))
                                      for _ in range(d)], dtype=torch.float64)
                        if self.spread_deg is not None else None)
            motion_b = (torch.tensor([math.radians(self._uniform(*self.motion_deg))
                                      * (1.0 if torch.rand(1, generator=self.rng) < 0.5
                                         else -1.0) for _ in range(d)],
                                     dtype=torch.float64)
                        if self.motion_deg is not None else None)
            Q_out[b] = Q
            alive = torch.ones(M, dtype=torch.complex128)
            if self.n_failed is not None:
                lo, hi = self.n_failed
                n_dead = int(torch.randint(lo, hi + 1, (1,), generator=self.rng))
                dead = 1 + torch.randperm(M - 1, generator=self.rng)[:n_dead]
                alive[dead] = 0.0
            if spread_b is not None:
                spread[b, :d] = spread_b
            if motion_b is not None:
                drift[b, :d] = motion_b
            imperfect.append(params.flatten())
            snr[b] = self._uniform(*self.snr_range)
            if self.per_source_freq:
                freqs[b, :d] = torch.tensor([self._uniform(*self.freq_range)
                                             for _ in range(d)], dtype=torch.float64)
            else:
                freqs[b] = self._uniform(*self.freq_range)
            corr_b = (self._uniform(*self.source_corr_range)
                      if self.source_corr_range is not None else self.source_corr)
            coherent[b] = corr_b > 0.99                                # label, not a mechanism
            nearfield[b] = self.range_range is not None
            # grid manifold at the design carrier for per-source draws (per-source manifolds
            # only exist AT the sources, built below)
            f_grid = 1.0 if self.per_source_freq else freqs[b]
            a_true_grid[b] = alive[:, None] * self.imperfections.true_manifold(geom, params,
                                                                               grid, f=f_grid)

            Kc = max(self.subcarriers, 1) if self.signal == 'ofdm' else 1
            bw = self.bandwidth if self.signal == 'ofdm' else 0.0
            for k in range(K):
                th = self._sample_doas(d, theta_range)
                doas[b, k, :d] = th
                if self.range_range is not None:                      # uniform distances
                    lo, hi = self.range_range
                    r = torch.tensor([self._uniform(lo, hi) for _ in range(d)],
                                     dtype=torch.float64)
                    ranges[b, k, :d] = r
                else:
                    r = None
                if self.per_source_freq:                          # per-column carrier steering
                    A = torch.cat([self.imperfections.true_manifold(
                        geom, params, th[i:i + 1], r=None if r is None else r[i:i + 1],
                        f=float(freqs[b, i]), gauge=False) for i in range(d)], dim=-1)
                else:
                    A = self.imperfections.true_manifold(geom, params, th, r=r, f=freqs[b],
                                                         gauge=False)
                if spread_b is not None:            # data lives on the SPREAD manifold
                    A = self._spread_steering(geom, params, th, float(freqs[b]), spread_b)
                if self.signal != 'gauss':          # representative steering: band center
                    fc = (freqs[b, :d].to(torch.float64) if self.per_source_freq
                          else torch.zeros(d, dtype=torch.float64))
                    A = self._broadband_steering(geom, params, th, fc + bw / 2)
                A = alive[:, None] * A
                a_true_doa[b, k, :, :d] = gauge_fix(A)

                Rs = self._source_cov(d, corr_b) * 10.0 ** (snr[b] / 10.0)
                R_s[b, k, :d, :d] = Rs
                L = torch.linalg.cholesky(Rs + 1e-10 * Rs.diagonal().real.mean()
                                          * torch.eye(d, dtype=torch.complex128))
                S = L @ torch.randn(d, T, generator=self.rng, dtype=torch.complex128)
                N = torch.randn(M, T, generator=self.rng, dtype=torch.complex128)
                if self.signal != 'gauss':          # line spectra: per-subcarrier steering
                    f_sub = (fc[:, None] + torch.arange(Kc, dtype=torch.float64)[None, :]
                             * (bw / Kc)).reshape(-1)                       # (d * Kc,)
                    A_sub = alive[:, None] * self._broadband_steering(
                        geom, params, th.repeat_interleave(Kc), f_sub)
                    w = Rs.diagonal().real.repeat_interleave(Kc) / Kc       # E|c|^2 per line
                    cr = torch.randn(d * Kc, 2, generator=self.rng, dtype=torch.float64)
                    c = torch.complex(cr[:, 0], cr[:, 1]) * (w / 2).sqrt().to(torch.complex128)
                    ph = torch.exp(1j * torch.pi * f_sub[:, None]
                                   * torch.arange(T, dtype=torch.float64)[None, :])
                    R_true[b, k] = torch.einsum('mi,i,ni->mn', A_sub,
                                                w.to(torch.complex128), A_sub.conj()) + Q
                    X[b, k, :, :T] = (A_sub @ (c[:, None] * ph)
                                      + (N if Lq is None else Lq @ N))
                elif motion_b is not None:          # truth reported at the window center
                    A_t = alive[:, None, None] * self._motion_steering(
                        geom, params, th, float(freqs[b]), motion_b, T)
                    R_true[b, k] = torch.einsum('mtd,de,nte->mn', A_t, Rs,
                                                A_t.conj()) / T + Q
                    X[b, k, :, :T] = (torch.einsum('mtd,dt->mt', A_t, S)
                                      + (N if Lq is None else Lq @ N))
                else:
                    R_true[b, k] = A @ Rs @ A.mH + Q
                    X[b, k, :, :T] = A @ S + (N if Lq is None else Lq @ N)

        R_hat = X @ X.mH / Ts[:, None, None, None].to(torch.float64)
        batch = {'X': X, 'R_hat': R_hat, 'R_true': R_true,
                'doas': doas, 'ranges': ranges, 'n_src': ds,
                'a_true_grid': a_true_grid, 'a_true_doa': a_true_doa, 'R_s': R_s,
                'sigma2': torch.ones(batch_size, dtype=torch.float64), 'snr': snr,
                'coherent': coherent, 'nearfield': nearfield, 'freqs': freqs,
                'snapshots': Ts, 'drift': drift, 'spread': spread, 'Q': Q_out,
                'positions': geom.positions.expand(batch_size, M, 3),
                'grid': grid, 'imperfect': torch.stack(imperfect)}
        return self._coarray(batch, geom) if self.coarray else batch

    #*************#
    #   coarray   #
    #*************#
    def _coarray_maps(self, geom):
        """Integer lag structure of a 1-D geometry: flat (i, j) -> lag index, redundancy
        counts, and the one-sided lag count L (virtual ULA size L + 1). Cached."""
        p = geom.positions[:, 0].round().long()
        if self._ca_pos is None or not torch.equal(self._ca_pos, p):
            diff = (p[:, None] - p[None, :]).reshape(-1)
            L = int(diff.max())
            counts = torch.zeros(2 * L + 1, dtype=torch.float64)
            counts.index_add_(0, diff + L, torch.ones_like(diff, dtype=torch.float64))
            assert (counts > 0).all(), 'coarray front-end needs a contiguous difference set'
            self._ca_pos, self._ca_idx, self._ca_counts, self._ca_L = p, diff + L, counts, L
        return self._ca_idx, self._ca_counts, self._ca_L

    def _coarray(self, batch, geom):
        """
        Sparse-array coarray front-end: average each covariance over equal sensor
        differences into the lag vector z (contiguous by construction), Hankel its L + 1
        windows into pseudo-snapshots, and hand every consumer a virtual (L + 1)-element
        ULA scene. Physical imperfections corrupt R BEFORE this map -- exactly the
        fragility the mismatch axis probes. True manifolds map through one-sided lag
        products (gauge-free: the pair product cancels any per-scene phase).
        """
        idx, counts, L = self._coarray_maps(geom)
        Mv, M = L + 1, self.M
        B = batch['X'].shape[0]
        win = torch.arange(Mv)[:, None] + torch.arange(Mv)[None, :]     # w_k[m] = z[m+k-L]

        def lag_avg(R):                                     # (..., M, M) -> (..., 2L+1)
            flat = R.reshape(-1, M * M)
            z = torch.zeros(flat.shape[0], 2 * L + 1, dtype=R.dtype)
            z.index_add_(1, idx, flat)
            return (z / counts).reshape(*R.shape[:-2], 2 * L + 1)

        def smooth(R):                                      # covariance -> virtual (Mv, Mv)
            W = lag_avg(R)[..., win]
            return W @ W.mH / Mv

        Xv = lag_avg(batch['R_hat'])[..., win]              # pseudo-snapshots (B, K, Mv, Mv)
        batch['X'] = Xv
        batch['R_hat'] = Xv @ Xv.mH / Mv
        batch['R_true'] = smooth(batch['R_true'])

        def virt(a):                                        # steering (..., M, n) -> (..., Mv, n)
            P = torch.einsum('...mg,...kg->...mkg', a, a.conj())
            flat = P.reshape(-1, M * M, a.shape[-1])
            z = torch.zeros(flat.shape[0], 2 * L + 1, a.shape[-1], dtype=a.dtype)
            z.index_add_(1, idx, flat)
            z = z / counts[:, None]
            return z[:, L:, :].reshape(*a.shape[:-2], Mv, a.shape[-1])

        batch['a_true_grid'] = virt(batch['a_true_grid'])
        batch['a_true_doa'] = virt(batch['a_true_doa'])
        batch['positions'] = ArrayGeometry.ula(Mv).positions.expand(B, Mv, 3)
        batch['snapshots'] = torch.full_like(batch['snapshots'], Mv)
        batch['Q'] = torch.eye(Mv, dtype=Xv.dtype).expand(B, Mv, Mv).clone()
        return batch
