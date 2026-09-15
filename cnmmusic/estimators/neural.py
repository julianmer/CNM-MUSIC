####################################################################################################
#                                            neural.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Learned estimators wrapped in the batched Estimator API so the evaluation harness       #
#          treats them exactly like the classical baselines.                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch

from cnmmusic.criteria.losses import subspace_ratio
from cnmmusic.estimators.base import Estimator


#**************************************************************************************************#
#                                        Class CNMEstimator                                        #
#**************************************************************************************************#
#                                                                                                  #
# The steering-correction CNM at test time: z from the observation, En from                        #
# the empirical covariance, then the model's own scan of the corrected null spectrum, joint over   #
# the active axes. The corrected manifold is a first-class function -- other back-ends (MVDR,      #
# ...) hook in through the same steer() call.                                                      #
#                                                                                                  #
#**************************************************************************************************#
class CNMEstimator(Estimator):
    spectrum_on_snapshots = True                    # spectrum() takes raw X, not a covariance

    def __init__(self, checkpoint, geom=None, config=None, method='music', gate=False,
                 subspace=None):
        super().__init__(geom, config)
        from cnmmusic.models.frameworkCNM import CNMFramework
        self.model = CNMFramework.load_from_checkpoint(checkpoint, map_location=self.device,
                                                       strict=False).eval()
        # optional SubspaceNet checkpoint: its learned surrogate covariance replaces the
        # sample covariance in the back-end (noise subspace / Capon inverse); the CNM
        # latent and manifold correction are unchanged
        self.ssn = None
        if subspace:
            from cnmmusic.models.frameworkBaselines import BaselinesFramework
            self.ssn = BaselinesFramework.load_from_checkpoint(subspace, map_location=self.device,
                                                               strict=False).eval()
        self.gate = gate                            # uncertainty-gated correction (gate_weight)
        self.sigma_ref = None                       # in-distribution spread, set by calibrate()
        # adopt the estimator's scan range (it may extend past the trained FOV so edge
        # nulls are whole); the model's native grid step is preserved
        hp = self.model.hparams
        if tuple(hp.theta_range) != tuple(self.theta_range):
            step = (hp.theta_range[1] - hp.theta_range[0]) / (hp.grid_size - 1)
            hp.grid_size = int(round((self.theta_range[1] - self.theta_range[0]) / step)) + 1
            hp.theta_range = tuple(self.theta_range)
        self.method = method                        # native corrected-null scan (music)
        # joint estimator contract for the eval harness: (doas, aux) when an axis is active
        self.joint_axis = ('range' if self.model.range_active
                           else 'freq' if self.model.freq_active else None)

    def _encode(self, X, R):
        # the framework's full latent path (representation -> backend), segment axis added
        return self.model.encode({'X': X[:, None], 'R_hat': R[:, None]})

    def _positions(self, B):
        return self.geom.positions[None].expand(B, -1, -1).to(torch.float64)

    def _backend_cov(self, X, R):
        """Covariance the back-end works on: the SubspaceNet surrogate if attached, else R."""
        return self.ssn.net(X).to(R.dtype) if self.ssn is not None else R

    @torch.no_grad()
    def spectrum(self, X, n_src=None):
        """Corrected-manifold null spectrum (B, G) on the plot grid — visualization only."""
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        R = self.covariance(X)
        z = self._encode(X, R)
        En = self.noise_subspace(self._backend_cov(X, R), n_src)
        B = X.shape[0]
        th = self.grid[None].expand(B, -1)
        J = self.model._scan_J(z, En, self._positions(B), th,
                               torch.full_like(th, torch.inf), torch.ones_like(th))
        P = 1.0 / (J + 1e-12)
        return P[0] if single else P

    #**********************#
    #   uncertainty gate   #
    #**********************#
    def _spread(self, z, positions, n_draws=16, kappa=100.0, n_probe=65):
        """
        Per-scene RMS spread of the correction Delta over tangent draws around the sphere
        direction (fixed kappa, as sample_doas), probed on a coarse far-field theta grid:
        -> sigma (B,). Dimensionless for 'mult', where Delta is relative to a0.
        """
        zd = self.model.hparams.z_dim
        mu, tok = z[..., :zd], z[..., zd:]
        th = torch.linspace(*self.theta_range, n_probe, dtype=torch.float64,
                            device=mu.device)[None].expand(z.shape[0], -1)
        uu, ff = torch.zeros_like(th), torch.ones_like(th)
        deltas = []
        for _ in range(n_draws):
            v = torch.randn_like(mu)
            v = v - (v * mu).sum(-1, keepdim=True) * mu                    # tangent noise
            zk = mu + v / (kappa ** 0.5)
            zk = zk / zk.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            deltas.append(self.model.correction(torch.cat([zk, tok], dim=-1),
                                                th, uu, ff, positions=positions))
        D = torch.stack(deltas)
        return (D - D.mean(0)).abs().pow(2).mean(dim=(0, 2, 3)).sqrt()

    @torch.no_grad()
    def scene_uncertainty(self, X):
        """Per-scene sphere spread sigma of the correction (see _spread)."""
        X, _ = self.batched(X)
        z = self._encode(X, self.covariance(X))
        return self._spread(z, self._positions(X.shape[0]))

    @torch.no_grad()
    def calibrate(self, X, q=0.9):
        """In-distribution reference spread: the q-quantile over calibration scenes X."""
        X, _ = self.batched(X)
        z = self._encode(X, self.covariance(X))
        self.sigma_ref = self._spread(z, self._positions(X.shape[0])).quantile(q).item()
        return self.sigma_ref

    def gate_weight(self, z, positions):
        """g = exp(-relu(sigma / sigma_ref - 1)): the full correction inside the calibrated
        in-distribution spread, exponential fallback toward the nominal physics beyond it."""
        assert self.sigma_ref is not None, 'gated estimator needs calibrate() first'
        sigma = self._spread(z, positions)
        return torch.exp(-(sigma / self.sigma_ref - 1.0).clamp_min(0.0))

    @torch.no_grad()
    def __call__(self, X, n_src=None):
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        if self.ssn is not None and X.shape[-1] <= self.ssn.hparams.tau:   # no tau-lag
            doas = torch.full((X.shape[0], n_src), torch.nan,             # autocorrelations
                              dtype=torch.float64, device=X.device)
            return doas[0] if single else doas
        R = self.covariance(X)
        z = self._encode(X, R)
        R = self._backend_cov(X, R)
        En = self.noise_subspace(R, n_src)
        pos = self._positions(X.shape[0])
        g = self.gate_weight(z, pos) if self.gate else None    # per-scene shrink toward a0
        steer_fn = lambda th, rr, ff: self.model.steer(z, pos, th, rr, ff, gate=g)
        spectrum = None
        if self.method == 'mvdr':                   # Capon on the corrected manifold (same
            eye = torch.eye(R.shape[-1], dtype=R.dtype, device=R.device)     # form as the
            load = 1e-3 * R.diagonal(dim1=-2, dim2=-1).real.mean(-1)         # mvdr val
            Rinv = torch.linalg.inv(R + load[..., None, None] * eye)         # back-end)

            def spectrum(th, rr, ff):
                a = steer_fn(th, rr, ff)
                a = a / a.abs().pow(2).sum(-2, keepdim=True).sqrt().clamp_min(1e-12)
                q = torch.einsum('bmn,bmk,bkn->bn', a.conj(), Rinv.to(a.dtype), a).real
                return 1.0 / q.clamp_min(1e-12)
        elif self.gate:
            spectrum = lambda th, rr, ff: 1.0 / (subspace_ratio(En, steer_fn(th, rr, ff))
                                                 + 1e-12)
        # the theta scan uses the estimator's grid (the SAME grid as every scanned classical
        # method), not the model's training grid
        doas, r_hat, f_hat = self.model.estimate(z, En, pos, n_src, spectrum=spectrum,
                                                 n_theta=self.grid_size)
        if self.joint_axis is not None:
            aux = r_hat if self.joint_axis == 'range' else f_hat
            return (doas[0], aux[0]) if single else (doas, aux)
        return doas[0] if single else doas


#**************************************************************************************************#
#                                   Class SubspaceNetEstimator                                     #
#**************************************************************************************************#
class SubspaceNetEstimator(Estimator):
    spectrum_on_snapshots = True                    # spectrum() takes raw X, not a covariance

    def __init__(self, checkpoint, geom=None, config=None, method='root'):
        super().__init__(geom, config)
        from cnmmusic.models.frameworkBaselines import BaselinesFramework
        self.model = BaselinesFramework.load_from_checkpoint(checkpoint,
                                                             map_location=self.device,
                                                             strict=False).eval()
        self.method = method

    @torch.no_grad()
    def spectrum(self, X, n_src=None):
        """MUSIC spectrum scanned on the learned surrogate covariance: X -> (B, G)."""
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        En = self.noise_subspace(self.model.net(X), n_src)
        A = self.A_grid.to(En.dtype)
        q = torch.einsum('bmk,mg->bkg', En.conj(), A).abs().pow(2).sum(-2)
        P = 1.0 / (q + 1e-12)
        return P[0] if single else P

    @torch.no_grad()
    def __call__(self, X, n_src=None):
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        if X.shape[-1] <= self.model.hparams.tau:   # the tau-lag autocorrelations do not exist
            doas = torch.full((X.shape[0], n_src), torch.nan,      # (training clamps T likewise)
                              dtype=torch.float64, device=X.device)
            return doas[0] if single else doas
        if self.method == 'music':                  # MUSIC back-end on the surrogate covariance
            doas = self.pick_peaks(self.spectrum(X, n_src), n_src)
        else:                                       # gridless Root-MUSIC back-end (default)
            doas = self.model.forward(X, n_src)
        return doas[0] if single else doas


#**************************************************************************************************#
#                                     Class DAMUSICEstimator                                       #
#**************************************************************************************************#
class DAMUSICEstimator(Estimator):
    def __init__(self, checkpoint, geom=None, config=None):
        super().__init__(geom, config)
        from cnmmusic.models.frameworkBaselines import BaselinesFramework
        self.model = BaselinesFramework.load_from_checkpoint(checkpoint,
                                                             map_location=self.device,
                                                             strict=False).eval()

    @torch.no_grad()
    def __call__(self, X, n_src=None):
        from cnmmusic.arrays.steering import steering_matrix
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        pos = self.geom.positions.to(self.device)
        A = steering_matrix(pos[None].to(torch.float64),
                            self.model.net.grid[None].to(self.device))[0]
        doas = self.model.net(X, n_src, A)[:, :n_src]
        return doas[0] if single else doas


#**************************************************************************************************#
#                                     Class GridCNNEstimator                                       #
#**************************************************************************************************#
class GridCNNEstimator(Estimator):
    def __init__(self, checkpoint, geom=None, config=None):
        super().__init__(geom, config)
        from cnmmusic.models.frameworkBaselines import BaselinesFramework
        self.model = BaselinesFramework.load_from_checkpoint(checkpoint,
                                                             map_location=self.device,
                                                             strict=False).eval()

    @torch.no_grad()
    def __call__(self, X, n_src=None):
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        doas = self.model.net.decode(self.model.net(X), n_src)
        return doas[0] if single else doas
#**************************************************************************************************#
#                                   Class CNMDAMUSICEstimator                                      #
#**************************************************************************************************#
#                                                                                                  #
# The corrected manifold under DA-MUSIC's back-end: DA-MUSIC's trained network forms its MUSIC     #
# spectrum from an EXPLICIT steering matrix, so CNM's per-scene corrected manifold drops straight  #
# in where the nominal one used to go. Spec: 'cnm-damusic:<cnm.ckpt>+<damusic.ckpt>'.              #
#                                                                                                  #
#**************************************************************************************************#
class CNMDAMUSICEstimator(Estimator):
    def __init__(self, checkpoints, geom=None, config=None):
        super().__init__(geom, config)
        cnm_ck, dam_ck = checkpoints.split('+')
        from cnmmusic.models.frameworkCNM import CNMFramework
        from cnmmusic.models.frameworkBaselines import BaselinesFramework
        self.cnm = CNMFramework.load_from_checkpoint(cnm_ck, map_location=self.device,
                                                     strict=False).eval()
        self.dam = BaselinesFramework.load_from_checkpoint(dam_ck, map_location=self.device,
                                                           strict=False).eval()

    @torch.no_grad()
    def __call__(self, X, n_src=None):
        n_src = n_src if n_src is not None else self.n_src
        X, single = self.batched(X)
        B = X.shape[0]
        R = self.covariance(X)
        z = self.cnm.encode({'X': X[:, None], 'R_hat': R[:, None]})
        pos = self.geom.positions[None].expand(B, -1, -1).to(torch.float64)
        th = self.dam.net.grid[None].expand(B, -1).to(pos.device)      # DA-MUSIC's own grid
        A = self.cnm.steer(z, pos, th, torch.full_like(th, torch.inf), torch.ones_like(th))
        doas = self.dam.net(X, n_src, A)[:, :n_src]
        return doas[0] if single else doas
