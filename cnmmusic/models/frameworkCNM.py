####################################################################################################
#                                         frameworkCNM.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 20/07/26                                                                                #
#                                                                                                  #
# Purpose: The conditional neural-manifold framework: an observation-conditioned correction of     #
#          the exact nominal steering manifold. Encoder h(X) is the input                          #
#          REPRESENTATION (snapshot GRU | covariance features | lag-covariance features), the      #
#          backend the only latent path after it, then a_hat(theta, r, f | z) =                    #
#          a0 * (1 + Delta(z, theta, u, f)) with a zero-init correction head, so the untrained     #
#          model is exactly classical physics and every downstream estimator falls back to its     #
#          textbook self. Training shapes the null landscape: J = ||En^H a||^2 / ||a||^2 is        #
#          driven to 0 at the true JOINT (theta, r, f) tuples and margin-ranked above the truth    #
#          at random negative tuples over the active axes. Two independent training switches       #
#          select the sample covariance R_hat or                                                   #
#          the exact ensemble covariance A R_s A^H + sigma2 I: train_cov for the covariance        #
#          encoder's input, train_En for the En in the loss; validation/inference always use the   #
#          sample estimate (train clean, deploy on the noisy approximation). Estimation scans the  #
#          corrected null spectrum jointly over the active axes; the corrected manifold is a       #
#          first-class function, so other back-ends (MVDR, ...) plug into the same steer()         #
#          call. The sphere backend provides inference-time vMF-style sampling (sample_doas)       #
#          for uncertainty.                                                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch

import torch.nn as nn

from cnmmusic.models.framework import Framework, sub_batch
from cnmmusic.archs.encoder import (CovarianceEncoder, SnapshotEncoder, LagCovarianceEncoder,
                                    SnapAttnEncoder, MeanSetEncoder)
from cnmmusic.archs.correction import SteeringCorrection
from cnmmusic.archs.music import DifferentiableRootMUSIC
from cnmmusic.arrays.steering import nearfield_steering_matrix
from cnmmusic.criteria.losses import (subspace_ratio, ranking_loss, likelihood_loss,
                                      landscape_kl_loss, likelihood_loss_beam)
from cnmmusic.criteria.metrics import matched_tuple_rmse


#**************************************************************************************************#
#                                        Class CNMFramework                                        #
#**************************************************************************************************#
class CNMFramework(Framework):
    def __init__(self, M=8, z_dim=16, hidden=128, encoder='snapshot', grid_size=720,
                 theta_range=(-1.047, 1.047), range_span=None, freq_span=(1.0, 1.0),
                 lr=1e-3, optimizer='adam', backend='none',
                 train_cov='empirical', train_En='empirical',
                 loss='rank', tau_nll=1.0, corr_mode='mult', pin_aux=False, eps_scale=1.0,
                 corr_cond='concat', coord_enc='harm',
                 n_neg=64, margin=2.0, excl=None, min_sep_rad=0.035, n_harm=4, tau=8,
                 n_probe=256, n_hard=0,
                 val_estimators=('music', 'mvdr', 'root', 'sbl'),
                 **kwargs):                              # tolerate factory-wide extras
        self.range_active = range_span is not None
        self.freq_active = tuple(freq_span) != (1.0, 1.0)
        if pin_aux:                       # aux axes pinned at NOMINAL (r = inf, f = f_c) in
            self.range_active = False     # loss and scan: the correction absorbs range /
            self.freq_active = False      # frequency effects like any other mismatch
        self.u_max = 1.0 / float(min(range_span)) if self.range_active else 0.0
        self.u_lo = 1.0 / float(max(range_span)) if self.range_active else 0.0
        self.f_halfspan = (max(abs(f - 1.0) for f in freq_span) if self.freq_active else 0.0)
        assert encoder in ('snapshot', 'covariance', 'lagcov', 'snapattn', 'meanset'), \
            f'unknown encoder: {encoder}'
        assert backend in ('none', 'mlp', 'sphere'), f'unknown backend: {backend}'
        assert train_cov in ('empirical', 'true'), f'unknown train_cov: {train_cov}'
        assert train_En in ('empirical', 'true'), f'unknown train_En: {train_En}'
        assert loss in ('rank', 'nll', 'nlls', 'ce', 'ce-beam'), f'unknown loss: {loss}'
        assert corr_mode in ('mult', 'add', 'free'), f'unknown corr_mode: {corr_mode}'
        assert corr_cond in ('concat', 'film', 'attn'), f'unknown corr_cond: {corr_cond}'
        assert coord_enc in ('harm', 'lff', 'raw'), f'unknown coord_enc: {coord_enc}'
        assert optimizer in ('adam', 'adamw'), f'unknown optimizer: {optimizer}'
        assert set(val_estimators) <= {'music', 'mvdr', 'root', 'sbl',
                                       'bbmusic', 'swmusic', 'gevd'}, \
            f'unknown val_estimators: {val_estimators}'
        super().__init__(M=M, z_dim=z_dim, hidden=hidden, encoder=encoder,
                         grid_size=grid_size,
                         theta_range=tuple(theta_range), range_span=range_span,
                         freq_span=tuple(freq_span), lr=lr, optimizer=optimizer,
                         backend=backend,
                         train_cov=train_cov, train_En=train_En,
                         loss=loss, tau_nll=tau_nll, corr_mode=corr_mode, pin_aux=pin_aux,
                         eps_scale=eps_scale,
                         corr_cond=corr_cond, coord_enc=coord_enc,
                         n_neg=n_neg, margin=margin,
                         excl=excl, min_sep_rad=min_sep_rad, n_harm=n_harm, tau=tau,
                         n_probe=n_probe, n_hard=n_hard,
                         val_estimators=tuple(val_estimators))
        self.encoder = (SnapshotEncoder(M=M, z_dim=z_dim, hidden=hidden)
                        if encoder == 'snapshot'
                        else CovarianceEncoder(M=M) if encoder == 'covariance'
                        else SnapAttnEncoder(M=M) if encoder == 'snapattn'
                        else MeanSetEncoder(M=M) if encoder == 'meanset'
                        else LagCovarianceEncoder(M=M, tau=tau))
        # backend 'none': the representation conditions the correction net DIRECTLY (covariance /
        # lag features un-mixed -- the correction MLP does all the work); mlp / sphere compress
        # to z_dim first
        self.latent_net = (nn.Sequential(nn.Linear(self.encoder.out_dim, hidden), nn.SiLU(),
                                         nn.Linear(hidden, z_dim))
                           if backend != 'none' else None)
        cond_dim = z_dim if backend != 'none' else self.encoder.out_dim
        self.correction = SteeringCorrection(M=M, z_dim=cond_dim, hidden=hidden, n_harm=n_harm,
                                             zero_init=(corr_mode != 'free'),
                                             cond_mode=corr_cond, coord_enc=coord_enc,
                                             theta_range=tuple(theta_range),
                                             u_span=((self.u_lo, self.u_max)
                                                     if self.range_active else None),
                                             f_span=(tuple(freq_span)
                                                     if self.freq_active else None),
                                             n_tok=M, d_tok=16)
        if corr_cond == 'attn':                  # per-sensor condition tokens straight from the
            self.token_head = nn.Linear(self.encoder.out_dim, M * 16)     # encoder features --
                                                                          # bypasses the z bottleneck
        self.root = DifferentiableRootMUSIC()          # 'root' validation back-end (angle-only)
        for axis, active in (('range', self.range_active), ('freq', self.freq_active)):
            if not active:
                print(f'[CNM] {axis} axis constant in this scenario -> pinned in the loss')

    #**************#
    #   encoding   #
    #**************#
    def _cov(self, batch, mode, k=0):
        """Sample or exact ensemble covariance; the exact one only ever in training mode."""
        key = 'R_true' if mode == 'true' and self.training else 'R_hat'
        return batch[key][:, k]

    @staticmethod
    def _lengths(batch, X):
        """Per-element snapshot counts (B,) for the masked encoders; None = full T."""
        Ts = batch.get('snapshots')
        if Ts is None:
            return None
        Ts = torch.as_tensor(Ts, device=X.device).reshape(-1)
        return Ts.expand(X.shape[0]) if Ts.numel() == 1 else Ts

    def encode(self, batch, k=0):
        if self.hparams.encoder == 'covariance':
            feats = self.encoder(self._cov(batch, self.hparams.train_cov, k))
        else:
            X = batch['X'][:, k]
            feats = self.encoder(X, self._lengths(batch, X))
        z = self.latent_net(feats) if self.latent_net is not None else feats
        if self.hparams.backend == 'sphere':
            z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        if self.hparams.corr_cond == 'attn':     # sensor tokens ride along after the sphere
            z = torch.cat([z, self.token_head(feats)], dim=-1)     # part; consumers split on
        return z                                                   # z_dim (correction, vMF)

    def _d_groups(self, batch):
        """
        Dense sub-batches grouped by SOURCE COUNT alone -- at most M - 1 groups per batch
        instead of one per (T, d) pair: the masked encoders handle mixed snapshot counts
        exactly, so only the source dimension still needs dense slicing. Yields
        (idx, sub_batch, d) with per-element 'snapshots' preserved for the length masks.
        """
        B = batch['X'].shape[0]
        Ts = torch.as_tensor(batch['snapshots']).reshape(-1)
        Ts = Ts.expand(B) if Ts.numel() == 1 else Ts
        ds = torch.as_tensor(batch['n_src']).reshape(-1)
        ds = ds.expand(B) if ds.numel() == 1 else ds
        for d in ds.unique().tolist():
            idx = (ds == int(d)).nonzero(as_tuple=True)[0]
            sub = sub_batch(batch, idx, int(Ts[idx].max()), int(d))
            sub['snapshots'] = Ts[idx]
            yield idx, sub, int(d)

    def forward(self, batch):
        return self.encode(batch)

    #************************#
    #   corrected manifold   #
    #************************#
    def steer(self, z, positions, theta, r, f, gate=None):
        """
        Corrected steering vectors at joint tuples: positions (B, M, 3), theta/r/f (B, n)
        (r = inf for far field, f per tuple) -> (B, M, n) complex. corr_mode 'mult' (default)
        is a0 * (1 + Delta), 'add' is a0 + Delta (both exactly a0 at zero-init); 'free' is
        Delta alone -- the unanchored ablation, standard init, no physics fallback. gate (B,)
        in [0, 1] scales Delta per scene toward the nominal physics (0 -> exactly a0 for
        'mult'/'add') -- the inference-time uncertainty fallback.
        """
        a0 = nearfield_steering_matrix(positions, theta, r, f)
        u = torch.where(torch.isfinite(r), 1.0 / r.clamp_min(1e-9), torch.zeros_like(r))
        delta = self.correction(z, theta, u, f, positions=positions).to(a0.dtype)
        if gate is not None:
            delta = delta * gate[:, None, None].to(a0.dtype)
        if self.hparams.corr_mode == 'mult':
            return a0 * (1.0 + delta)
        if self.hparams.corr_mode == 'add':
            return a0 + delta
        return delta

    def _scan_J(self, z, En, positions, theta, r, f, chunk=4096):
        """subspace_ratio over many tuples, chunked so scan grids never blow up memory."""
        out = []
        for s in range(0, theta.shape[-1], chunk):
            a = self.steer(z, positions, theta[..., s:s + chunk], r[..., s:s + chunk],
                           f[..., s:s + chunk])
            out.append(subspace_ratio(En, a))
        return torch.cat(out, dim=-1)

    @staticmethod
    def _scan_P(fn, theta, r, f, chunk=4096):
        """
        Chunked evaluation of a spectrum callback fn(theta, r, f) -> P over many tuples.
        Callbacks with fn.global_spectrum = True operate on the whole grid jointly (SBL
        iterates over the full dictionary) and are called once, unchunked.
        """
        if getattr(fn, 'global_spectrum', False):
            return fn(theta, r, f)
        out = []
        for s in range(0, theta.shape[-1], chunk):
            out.append(fn(theta[..., s:s + chunk], r[..., s:s + chunk], f[..., s:s + chunk]))
        return torch.cat(out, dim=-1)

    #*********************#
    #   noise subspaces   #
    #*********************#
    @staticmethod
    def noise_subspace(R, n_src):
        """En (..., M, M - n_src); detached -- it comes from the data, not the model."""
        _, evecs = torch.linalg.eigh(R)
        return evecs[..., :R.shape[-1] - n_src].detach()

    #************************************************#
    #   candidate exclusion (rivals vs non-rivals)   #
    #************************************************#
    @staticmethod
    def _overlap_deficit(A, B):
        """Gauge-invariant nominal-manifold distance 1 - |a^H b|^2 / (||a||^2 ||b||^2) between
        all pairs of steering-vector columns: A (B, M, d), B (B, M, n) -> (B, d, n)."""
        num = torch.einsum('bmd,bmn->bdn', A.conj(), B).abs().pow(2)
        den = (A.abs().pow(2).sum(-2)[..., :, None]
               * B.abs().pow(2).sum(-2)[..., None, :]).clamp_min(1e-30)
        return 1.0 - num / den

    def _excl_eps(self, positions):
        """Auto-calibrated threshold: the overlap deficit of two broadside far-field steering
        vectors min_sep / 2 apart (per scene geometry) -- the parameter ring's radius,
        translated into the manifold metric once, so no per-axis constants exist."""
        th = torch.tensor([0.0, self.hparams.min_sep_rad / 2], dtype=torch.float64,
                          device=positions.device)[None].expand(positions.shape[0], 2)
        rr = torch.full_like(th, torch.inf)
        a = nearfield_steering_matrix(positions, th, rr, torch.ones_like(th))
        eps = self._overlap_deficit(a[..., :1], a[..., 1:])[:, 0, :]       # (B, 1)
        return eps * self.hparams.get('eps_scale', 1.0)   # 1.0 = the min_sep/2 ring; smaller
                                                          # = a target bump closer to the
                                                          # data resolution

    def _excl_mask(self, positions, doas, r_true, f_true, th, rr, ff):
        """
        True where a candidate tuple is NOT a rival of any truth. hp.excl = None: MANIFOLD
        metric -- a candidate whose nominal steering vector is closer to a truth's than the
        auto eps is indistinguishable to the data (covers the theta resolution cell, far-field
        range blindness, and exact ambiguities like the ULA f*sin(theta) ridge, where the
        deficit is 0). The mask anchors on a0, never on the learned manifold (ungameable).
        hp.excl = float: legacy span-normalized parameter-space radius.
        """
        hp = self.hparams
        if hp.excl is None:
            A_t = nearfield_steering_matrix(positions, doas, r_true, f_true)
            A_c = nearfield_steering_matrix(positions, th, rr, ff)
            deficit = self._overlap_deficit(A_t, A_c)                       # (B, d, n)
            return deficit.min(dim=1).values < self._excl_eps(positions)
        lo, hi = hp.theta_range
        u_true = torch.where(torch.isfinite(r_true), 1.0 / r_true.clamp_min(1e-9),
                             torch.zeros_like(r_true))
        u_c = torch.where(torch.isfinite(rr), 1.0 / rr.clamp_min(1e-9), torch.zeros_like(rr))
        dist = ((doas[..., None] - th[:, None, :]) / (hi - lo)).pow(2)
        if self.range_active:
            dist = dist + ((u_true[..., None] - u_c[:, None, :]) / self.u_max).pow(2)
        if self.freq_active:
            dist = dist + ((f_true[..., None] - ff[:, None, :])
                           / (2 * self.f_halfspan)).pow(2)
        return dist.min(dim=1).values < hp.excl ** 2

    #************#
    #   losses   #
    #************#
    def _tuples(self, batch, d):
        """True joint tuples of one dense sub-batch: doas/r (B, d), per-source f (B, d)."""
        doas = batch['doas'][:, 0]
        r_true = batch['ranges'][:, 0]
        f_true = (batch['freqs'][:, :d] if batch['freqs'].ndim == 2
                  else batch['freqs'][:, None].expand_as(doas))
        if self.hparams.get('pin_aux', False):     # truths pinned at nominal too: the loss
            r_true = torch.full_like(r_true, torch.inf)     # anchors the corrected NOMINAL
            f_true = torch.ones_like(f_true)                # manifold on the true angles
        return doas, r_true, f_true

    def _losses_core(self, batch):
        hp = self.hparams
        z = self.encode(batch, k=0)
        d = batch['n_src']
        B = batch['X'].shape[0]
        pos = batch['positions']
        doas, r_true, f_true = self._tuples(batch, d)
        u_true = torch.where(torch.isfinite(r_true), 1.0 / r_true.clamp_min(1e-9),
                             torch.zeros_like(r_true))
        En = self.noise_subspace(self._cov(batch, hp.train_En), d)

        # orthogonality at the true JOINT tuples
        J_true = subspace_ratio(En, self.steer(z, pos, doas, r_true, f_true))       # (B, d)

        # negatives: random tuples over the active axes
        lo, hi = hp.theta_range
        th_n = lo + (hi - lo) * torch.rand(B, hp.n_neg, dtype=doas.dtype,
                                           device=doas.device)
        if self.range_active:
            u_n = self.u_lo + (self.u_max - self.u_lo) * torch.rand_like(th_n)
            r_n = 1.0 / u_n.clamp_min(1e-9)
        else:
            r_n = torch.full_like(th_n, torch.inf)
        if self.freq_active:
            flo, fhi = hp.freq_span
            f_n = flo + (fhi - flo) * torch.rand_like(th_n)
        else:
            f_n = torch.ones_like(th_n)

        if hp.n_hard > 0:            # hard-negative mining: probe the CURRENT landscape and
            th_p = lo + (hi - lo) * torch.rand(B, hp.n_probe, dtype=doas.dtype,   # add its
                                               device=doas.device)  # deepest spurious nulls
            if self.range_active:
                u_p = self.u_lo + (self.u_max - self.u_lo) * torch.rand_like(th_p)
                r_p = 1.0 / u_p.clamp_min(1e-9)
            else:
                r_p = torch.full_like(th_p, torch.inf)
            if self.freq_active:
                flo, fhi = hp.freq_span
                f_p = flo + (fhi - flo) * torch.rand_like(th_p)
            else:
                f_p = torch.ones_like(th_p)
            with torch.no_grad():
                J_p = subspace_ratio(En, self.steer(z, pos, th_p, r_p, f_p))
                J_p = torch.where(self._excl_mask(pos, doas, r_true, f_true,   # truth cells /
                                                  th_p, r_p, f_p),            # non-rivals are
                                  torch.ones_like(J_p), J_p)                  # never selected
                idx = J_p.topk(hp.n_hard, dim=-1, largest=False).indices
            th_n = torch.cat([th_n, th_p.gather(-1, idx)], dim=-1)
            r_n = torch.cat([r_n, r_p.gather(-1, idx)], dim=-1)
            f_n = torch.cat([f_n, f_p.gather(-1, idx)], dim=-1)

        J_neg = subspace_ratio(En, self.steer(z, pos, th_n, r_n, f_n))

        l_ortho = J_true.mean()
        if hp.loss == 'ce':
            # plain one-hot cross-entropy, no exclusion: cell neighbours and ambiguity sets
            # converge to the label posterior by themselves -- sharp where the data is sharp
            l_rank = likelihood_loss(J_true, J_neg, tau=hp.tau_nll)
        elif hp.loss == 'ce-beam':
            A_t0 = nearfield_steering_matrix(pos, doas, r_true, f_true)
            A_c0 = nearfield_steering_matrix(pos, th_n, r_n, f_n)
            overlap = 1.0 - self._overlap_deficit(A_t0, A_c0)              # (B, d, n)
            l_rank = likelihood_loss_beam(J_true, J_neg, overlap, tau=hp.tau_nll)
        elif hp.loss == 'nlls':
            A_t0 = nearfield_steering_matrix(pos, doas, r_true, f_true)
            A_c0 = nearfield_steering_matrix(pos, th_n, r_n, f_n)
            deficit = self._overlap_deficit(A_t0, A_c0)                    # (B, d, n)
            if self.freq_active:                 # the record distinguishes along the f axis
                Ts = torch.as_tensor(batch['snapshots'],                   # what the array
                                     device=deficit.device).reshape(-1)    # alone cannot: the
                x = 0.5 * torch.pi * (f_true[..., :, None] - f_n[..., None, :])
                T_ = Ts[:, None, None].to(deficit.dtype)                   # temporal overlap is
                dir2 = torch.where(x.abs() < 1e-9, torch.ones_like(x),     # the record-length
                                   torch.sin(T_ * x) / (T_ * torch.sin(x))).pow(2)
                deficit = 1.0 - (1.0 - deficit) * dir2                     # Dirichlet kernel
            eps = self._excl_eps(pos)[..., None]                           # (B, 1, 1)
            q_target = torch.exp(-deficit / eps).sum(dim=1)                # ideal spectrum:
            l_rank = landscape_kl_loss(J_neg, q_target, tau=hp.tau_nll)    # one bump per truth
        elif hp.loss == 'nll':
            # exclusion: candidates the data cannot distinguish from a truth are no rival
            # hypotheses -- masked out of the candidate set (hinge below: neutralized, J := 1)
            excl_mask = self._excl_mask(pos, doas, r_true, f_true, th_n, r_n, f_n)
            l_rank = likelihood_loss(J_true, J_neg, tau=hp.tau_nll, excl_mask=excl_mask)
        else:
            excl_mask = self._excl_mask(pos, doas, r_true, f_true, th_n, r_n, f_n)
            J_neg = torch.where(excl_mask, torch.ones_like(J_neg), J_neg)
            l_rank = ranking_loss(J_true, J_neg, margin=hp.margin)
        loss = l_ortho + l_rank
        return loss, {'l_ortho': l_ortho, 'l_rank': l_rank, 'loss': loss}

    def compute_losses(self, batch):
        """Batches mix T and d per element: run the loss core per dense source-count group
        (the masked encoders absorb the mixed snapshot counts)."""
        total, logs = 0.0, {}
        B = batch['X'].shape[0]
        for idx, sub, _ in self._d_groups(batch):
            w = len(idx) / B
            loss_g, logs_g = self._losses_core(sub)
            total = total + w * loss_g
            for key, v in logs_g.items():
                logs[key] = logs.get(key, 0.0) + w * v
        logs['loss'] = total
        return total, logs

    #**************************************#
    #   estimation (corrected null scan)   #
    #**************************************#
    @staticmethod
    def _pick_1d(P, grid, d):
        """Top-d interior local maxima with parabolic refinement -> (B, d) sorted angles."""
        B, G = P.shape
        interior = (P[:, 1:-1] >= P[:, :-2]) & (P[:, 1:-1] >= P[:, 2:])
        vals = torch.where(interior, P[:, 1:-1], torch.full_like(P[:, 1:-1], -torch.inf))
        top_v, top_i = vals.topk(d, dim=-1)
        idx = top_i + 1
        fallback = P.topk(d, dim=-1).indices.clamp(1, G - 2)    # degenerate spectra
        idx = torch.where(torch.isfinite(top_v), idx, fallback)
        y0, y1, y2 = P.gather(1, idx - 1), P.gather(1, idx), P.gather(1, idx + 1)
        denom = y0 - 2 * y1 + y2
        off = torch.where(denom.abs() > 1e-18, 0.5 * (y0 - y2) / denom,
                          torch.zeros_like(denom))
        doas = grid[idx] + off.clamp(-1.0, 1.0) * (grid[1] - grid[0])
        return doas.sort(dim=-1).values

    @staticmethod
    def _pick_2d(P, grid, r_grid, d, excl_deg=4.0):
        """
        Iterative 2-D peak extraction (angular-column suppression after each pick: a close
        source's (theta, r) ridge is elongated in range, so plain top-d maxima would stack
        picks on one ridge). Returns (doas (B, d), ranges (B, d)) sorted by angle.
        """
        B, R, G = P.shape
        excl = int(excl_deg / torch.rad2deg(grid[1] - grid[0]).item()) + 1
        work = P.clone().reshape(B, -1)
        doas = torch.empty(B, d, dtype=grid.dtype, device=grid.device)
        rngs = torch.empty(B, d, dtype=r_grid.dtype, device=grid.device)
        cols = torch.arange(G, device=grid.device)
        for k in range(d):
            idx = work.argmax(dim=-1)
            g = idx % G
            doas[:, k] = grid[g]
            rngs[:, k] = r_grid[idx // G]
            lo = (g - excl).clamp(0)
            hi = (g + excl).clamp(max=G - 1)
            mask = (cols[None, :] >= lo[:, None]) & (cols[None, :] <= hi[:, None])
            work = work.reshape(B, R, G).masked_fill(mask[:, None, :], 0.0).reshape(B, -1)
        order = doas.argsort(dim=-1)
        return doas.gather(1, order), rngs.gather(1, order)

    @torch.no_grad()
    def estimate(self, z, En, positions, d, n_theta=None, n_u=None, n_f=None, chunk=4096,
                 spectrum=None):
        """
        Joint estimates from a scanned spectrum: a joint (theta, r) or (theta, f) 2-D scan
        when a second axis is active (theta alone otherwise), then a per-source frequency
        readout at the estimated (theta, r) when BOTH extra axes are active. Joint scanning
        matters: a sequential theta-then-f readout sits on the f sin(theta) ambiguity ridge
        of linear arrays; note (theta, f) is only identifiable at all for non-linear
        geometries. `spectrum` is an optional callback (theta, r, f) -> P (B, n) replacing
        the corrected null spectrum 1 / J -- the hook other back-ends (MVDR, ...) plug into.
        Returns (doas (B, d), ranges | None, freqs | None), each (B, d).
        """
        fn = spectrum if spectrum is not None else (
            lambda th, rr, ff: 1.0 / (subspace_ratio(En, self.steer(z, positions, th, rr, ff))
                                      + 1e-12))
        hp = self.hparams
        n_theta = n_theta or hp.grid_size        # ONE shared grid size for every scanned axis
        n_u = n_u or hp.grid_size
        n_f = n_f or hp.grid_size
        B = z.shape[0]
        lo, hi = hp.theta_range
        grid = torch.linspace(lo, hi, n_theta, dtype=torch.float64, device=z.device)
        flo, fhi = hp.freq_span
        f_grid = torch.linspace(flo, fhi, n_f, dtype=torch.float64, device=z.device)
        if self.range_active:                                  # joint (theta, r) at f = 1
            r_lo, r_hi = hp.range_span
            r_grid = torch.logspace(torch.log10(torch.tensor(float(r_lo))),
                                    torch.log10(torch.tensor(float(r_hi))), n_u,
                                    dtype=torch.float64, device=z.device)
            th = grid.repeat(n_u)
            rr = r_grid.repeat_interleave(n_theta)
            ff = torch.ones_like(th)
        elif self.freq_active:                                 # joint (theta, f), far field
            th = grid.repeat(n_f)
            rr = torch.full_like(th, torch.inf)
            ff = f_grid.repeat_interleave(n_theta)
        else:                                                  # theta alone
            th = grid
            rr = torch.full_like(th, torch.inf)
            ff = torch.ones_like(th)
        P = self._scan_P(fn, th[None].expand(B, -1), rr[None].expand(B, -1),
                         ff[None].expand(B, -1), chunk)
        r_hat, f_hat = None, None
        if self.range_active:
            doas, r_hat = self._pick_2d(P.reshape(B, n_u, n_theta), grid, r_grid, d)
        elif self.freq_active:
            doas, f_hat = self._pick_2d(P.reshape(B, n_f, n_theta), grid, f_grid, d)
        else:
            doas = self._pick_1d(P, grid, d)

        if self.freq_active and self.range_active:             # f readout at (theta, r)
            th_f = doas.repeat_interleave(n_f, dim=-1)                     # (B, d * n_f)
            rr_f = r_hat.repeat_interleave(n_f, dim=-1)
            ff_f = f_grid.repeat(d)[None].expand(B, -1)
            Pf = self._scan_P(fn, th_f, rr_f, ff_f, chunk).reshape(B, d, n_f)
            idx = Pf.argmax(dim=-1).clamp(1, n_f - 2)
            y0 = Pf.gather(-1, (idx - 1)[..., None])[..., 0]
            y1 = Pf.gather(-1, idx[..., None])[..., 0]
            y2 = Pf.gather(-1, (idx + 1)[..., None])[..., 0]
            denom = y0 - 2 * y1 + y2
            off = torch.where(denom.abs() > 1e-18, 0.5 * (y0 - y2) / denom,
                              torch.zeros_like(denom))
            f_hat = f_grid[idx] + off.clamp(-1.0, 1.0) * (f_grid[1] - f_grid[0])
        return doas, r_hat, f_hat

    #**************************************#
    #   sphere sampling (inference only)   #
    #**************************************#
    @torch.no_grad()
    def sample_doas(self, batch, n_samples=16, kappa=100.0):
        """
        Sphere backend only: K latent draws around the mean direction (tangent-Gaussian
        approximation of vMF with concentration kappa) -> K corrected manifolds -> K scanned
        estimates (K, B, d). The spread is the parameter uncertainty; kappa is NOT trained --
        calibrate it post-hoc on validation coverage.
        """
        assert self.hparams.backend == 'sphere', 'sampling needs the sphere backend'
        cond = self.encode(batch)
        zd = self.hparams.z_dim
        mu, tok = cond[..., :zd], cond[..., zd:]   # only the sphere part is a vMF direction;
        d = int(torch.as_tensor(batch['n_src']).reshape(-1)[0])   # attn tokens stay fixed
        En = self.noise_subspace(batch['R_hat'][:, 0], d)
        out = []
        for _ in range(n_samples):
            v = torch.randn_like(mu)
            v = v - (v * mu).sum(-1, keepdim=True) * mu                    # tangent noise
            z = mu + v / (kappa ** 0.5)
            z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            out.append(self.estimate(torch.cat([z, tok], dim=-1), En,
                                     batch['positions'], d)[0])
        return torch.stack(out)

    #*************************#
    #   lightning interface   #
    #*************************#
    def training_step(self, batch, batch_idx):
        loss, logs = self.compute_losses(batch)
        self.log_dict({f'train/{k}': v for k, v in logs.items()}, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.compute_losses(batch)
        self.log_dict({f'val/{k}': v for k, v in logs.items()}, on_epoch=True)
        self.log('val_loss', logs['loss'], on_epoch=True)
        self.log_estimation(batch)
        return loss

    def _val_spectra(self, z, En, R, positions, d):
        """
        The enabled (hparams.val_estimators) scan back-ends as spectrum callbacks for
        estimate(). Corrected manifold: 'music/' (null-spectrum scan; None -> estimate()'s
        default), 'mvdr/' (MVDR on the diagonally loaded sample covariance), and 'sbl/'
        (multisnapshot sparse Bayesian learning, Gerstoft et al., IEEE SPL 2016: the SBL1
        fixed point on the source-power hyperparameters gamma over the full dictionary,
        noise variance from the stochastic-ML estimate tr[(I-P)R]/(M-d); gamma is the
        spectrum). Steering is unit-normalized throughout so the near-field amplitude taper
        does not skew any spectrum. Nominal-manifold references 'music0/', 'mvdr0/',
        'sbl0/': the same estimators on a0 -- the corrected-vs-nominal gap isolates what
        the correction contributes.
        """
        M = R.shape[-1]
        eye = torch.eye(M, dtype=R.dtype, device=R.device)
        load = 1e-3 * R.diagonal(dim1=-2, dim2=-1).real.mean(-1)
        Rinv = torch.linalg.inv(R + load[..., None, None] * eye)

        def corrected(th, rr, ff):
            return self.steer(z, positions, th, rr, ff)

        def nominal(th, rr, ff):
            return nearfield_steering_matrix(positions, th, rr, ff)

        def capon(a):
            a = a / a.abs().pow(2).sum(-2, keepdim=True).sqrt().clamp_min(1e-12)
            q = torch.einsum('bmn,bmk,bkn->bn', a.conj(), Rinv.to(a.dtype), a).real
            return 1.0 / q.clamp_min(1e-12)

        def mvdr(th, rr, ff):
            return capon(corrected(th, rr, ff))

        def mvdr0(th, rr, ff):
            return capon(nominal(th, rr, ff))

        def music0(th, rr, ff):
            return 1.0 / (subspace_ratio(En, nominal(th, rr, ff)) + 1e-12)

        def sbl_factory(steer_fn, iters=40):
            def sbl(th, rr, ff):
                a = torch.cat([steer_fn(th[..., s:s + 4096], rr[..., s:s + 4096],
                                        ff[..., s:s + 4096])
                               for s in range(0, th.shape[-1], 4096)], dim=-1)
                a = a / a.abs().pow(2).sum(-2, keepdim=True).sqrt().clamp_min(1e-12)
                Rc = R.to(a.dtype)
                gamma = torch.einsum('bmg,bmk,bkg->bg', a.conj(), Rc, a).real.clamp_min(1e-12)
                sig = 0.1 * R.diagonal(dim1=-2, dim2=-1).real.mean(-1)
                for _ in range(iters):
                    Sig = (a * gamma[:, None, :].to(a.dtype)) @ a.mH \
                        + sig[:, None, None].to(a.dtype) * eye
                    t = torch.linalg.solve(Sig, a)                        # Sigma^-1 a
                    num = (t.conj() * (Rc @ t)).sum(-2).real              # a^H S^-1 R S^-1 a
                    den = (a.conj() * t).sum(-2).real.clamp_min(1e-30)    # a^H S^-1 a
                    gamma = gamma * (num / den).clamp_min(0.0).sqrt()     # SBL1 fixed point
                    idx = gamma.topk(d, dim=-1).indices                   # active set (K = d)
                    Am = a.gather(-1, idx[:, None, :].expand(-1, M, -1))
                    gram = Am.mH @ Am + 1e-9 * torch.eye(d, dtype=a.dtype, device=a.device)
                    Pp = Am @ torch.linalg.solve(gram, Am.mH)
                    sig = (torch.einsum('bii->b', (eye - Pp) @ Rc).real
                           / (M - d)).clamp_min(1e-6)
                return gamma
            sbl.global_spectrum = True
            return sbl

        specs = {}
        if 'music' in self.hparams.val_estimators:
            specs.update({'music/': None, 'music0/': music0})
        if 'mvdr' in self.hparams.val_estimators:
            specs.update({'mvdr/': mvdr, 'mvdr0/': mvdr0})
        if 'sbl' in self.hparams.val_estimators:
            specs.update({'sbl/': sbl_factory(corrected), 'sbl0/': sbl_factory(nominal)})
        return specs

    #********************************************************#
    #   scenario-specific classical baselines (theta-only)   #
    #********************************************************#
    @torch.no_grad()
    def _bbmusic_doas(self, X, Ts, positions, d, grid):
        """Incoherent broadband MUSIC (DA-MUSIC baseline): FFT the record, group the positive
        bins, per-group covariance -> EVD -> nominal pseudospectrum at the group's center
        carrier, AVERAGE the spectra over groups, pick d peaks."""
        B, M, _ = X.shape
        T = int(Ts.min())
        Xf = torch.fft.fft(X[..., :T], dim=-1)
        half = T // 2
        G = max(1, min(16, half // (2 * M)))
        P = torch.zeros(B, grid.shape[0], dtype=torch.float64, device=X.device)
        rr = torch.full((B, grid.shape[0]), torch.inf, dtype=torch.float64, device=X.device)
        for g in range(G):
            lo = 1 + g * max(half - 1, 0) // G
            hi = 1 + (g + 1) * max(half - 1, 0) // G
            if hi <= lo:
                continue
            bins = Xf[..., lo:hi]
            En = self.noise_subspace(bins @ bins.mH / (hi - lo), min(d, M - 1))
            A_g = nearfield_steering_matrix(positions, grid[None].expand(B, -1), rr,
                                            torch.full_like(rr, (lo + hi) / T))
            P = P + 1.0 / (subspace_ratio(En, A_g) + 1e-12)
        return self._pick_1d(P, grid, d)

    @torch.no_grad()
    def _gevd_doas(self, R, Qn, positions, d, grid):
        """Known-Q oracle (Schmidt): whiten with the TRUE noise covariance, classical MUSIC
        on the whitened pencil -- the upper reference for colored-noise scenes."""
        Li = torch.linalg.inv(torch.linalg.cholesky(Qn))
        En = self.noise_subspace(Li @ R @ Li.mH, d)
        B = R.shape[0]
        rr = torch.full((B, grid.shape[0]), torch.inf, dtype=torch.float64, device=R.device)
        A = nearfield_steering_matrix(positions, grid[None].expand(B, -1), rr,
                                      torch.ones_like(rr))
        P = 1.0 / (subspace_ratio(En, Li @ A.to(Li.dtype)) + 1e-12)
        return self._pick_1d(P, grid, d)

    @torch.no_grad()
    def _swmusic_errs(self, sub, z, d, grid):
        """Sub-window MUSIC on the corrected ('swmusic/') and nominal ('swmusic0/') manifolds:
        classical estimation per time window, scored against the linearly drifting truth at
        each window center -- the short-window answer to moving sources."""
        X = sub['X'][:, 0]
        B, M, _ = X.shape
        T = int(torch.as_tensor(sub['snapshots']).reshape(-1).min())
        W = max(1, min(8, T // (4 * M)))
        lo, hi = self.hparams.theta_range
        doas0 = sub['doas'][:, 0]
        dr = sub['drift'][:, :d] if 'drift' in sub else torch.zeros_like(doas0)
        rr = torch.full((B, grid.shape[0]), torch.inf, dtype=torch.float64, device=X.device)
        ff = torch.ones_like(rr)
        A_nom = nearfield_steering_matrix(sub['positions'], grid[None].expand(B, -1), rr, ff)
        A_cor = self.steer(z, sub['positions'], grid[None].expand(B, -1), rr, ff)
        errs_c, errs_n = [], []
        for wi in range(W):
            sl = slice(wi * T // W, (wi + 1) * T // W)
            Xw = X[..., sl]
            En = self.noise_subspace(Xw @ Xw.mH / max(sl.stop - sl.start, 1), d)
            truth_w = doas0 + dr * (((sl.start + sl.stop) / 2) / T - 0.5)
            for A, out in ((A_cor, errs_c), (A_nom, errs_n)):
                est = self._pick_1d(1.0 / (subspace_ratio(En, A) + 1e-12), grid, d)
                out.append(matched_tuple_rmse(est, truth_w, [], theta_span=hi - lo)[0])
        return torch.cat(errs_c), torch.cat(errs_n)

    def _calibrate(self, R, z):
        """
        The broadside part of the correction (theta = 0, far field, carrier) folded out of
        the covariance -- the only part a Vandermonde rooting can absorb; everything
        tuple-dependent is structurally invisible to the root polynomial.
        """
        th0 = torch.zeros(z.shape[0], 1, dtype=torch.float64, device=z.device)
        c = 1.0 + self.correction(z, th0, torch.zeros_like(th0),
                                  torch.ones_like(th0))[..., 0]        # (B, M)
        c = c / c.abs().clamp_min(1e-12) * c.abs().clamp_min(1e-3)
        Dinv = torch.diag_embed(1.0 / c).to(R.dtype)
        return Dinv @ R @ Dinv.mH

    @torch.no_grad()
    def estimation_metrics(self, batch):
        """
        Deployment metrics on one batch: every validation back-end plugs the corrected
        manifold into its estimator on the sample covariance, scored with ONE joint
        permutation per scene (matched_tuple_rmse) so ghost pairings are punished, not
        hidden by per-axis matching. Returns {metric_name: value}.
        """
        lo, hi = self.hparams.theta_range
        errs = {}
        for _, sub, d in self._d_groups(batch):
            z = self.encode(sub, k=0)
            R = sub['R_hat'][:, 0]
            En = self.noise_subspace(R, d)
            doas, r_true, f_true = self._tuples(sub, d)
            u_t = torch.where(torch.isfinite(r_true), 1.0 / r_true.clamp_min(1e-9),
                              torch.zeros_like(r_true))
            for prefix, fn in self._val_spectra(z, En, R, sub['positions'], d).items():
                doas_e, r_e, f_e = self.estimate(z, En, sub['positions'], d, spectrum=fn)
                aux = []
                if r_e is not None:
                    aux.append((1.0 / r_e.clamp_min(1e-9), u_t, self.u_max, r_e, r_true))
                if f_e is not None:
                    aux.append((f_e, f_true, 2 * self.f_halfspan))
                th, ax = matched_tuple_rmse(doas_e, doas, aux, theta_span=hi - lo)
                errs.setdefault((prefix, 'theta'), []).append(th)
                k = 0
                if r_e is not None:
                    errs.setdefault((prefix, 'range'), []).append(ax[k])
                    k += 1
                if f_e is not None:
                    errs.setdefault((prefix, 'freq'), []).append(ax[k])
            if 'root' in self.hparams.val_estimators:              # angle-only rooting: the
                for prefix, zz in (('root/', z), ('root0/', None)):    # far-field ULA null
                    En_r = (En if zz is None                           # polynomial on the
                            else self.noise_subspace(self._calibrate(R, zz), d))  # (calibrated)
                    th, _ = matched_tuple_rmse(self.root(En_r, d), doas, [],    # covariance
                                               theta_span=hi - lo)
                    errs.setdefault((prefix, 'theta'), []).append(th)
            grid_t = torch.linspace(lo, hi, self.hparams.grid_size, dtype=torch.float64,
                                    device=R.device)
            if 'bbmusic' in self.hparams.val_estimators:
                est = self._bbmusic_doas(sub['X'][:, 0],
                                         torch.as_tensor(sub['snapshots']).reshape(-1),
                                         sub['positions'], d, grid_t)
                th, _ = matched_tuple_rmse(est, doas, [], theta_span=hi - lo)
                errs.setdefault(('bbmusic0/', 'theta'), []).append(th)
            if 'gevd' in self.hparams.val_estimators and 'Q' in sub:
                est = self._gevd_doas(R, sub['Q'], sub['positions'], d, grid_t)
                th, _ = matched_tuple_rmse(est, doas, [], theta_span=hi - lo)
                errs.setdefault(('gevd0/', 'theta'), []).append(th)
            if 'swmusic' in self.hparams.val_estimators and not self.range_active \
                    and not self.freq_active:
                ec, en0 = self._swmusic_errs(sub, z, d, grid_t)
                errs.setdefault(('swmusic/', 'theta'), []).append(ec)
                errs.setdefault(('swmusic0/', 'theta'), []).append(en0)
        out, norm = {}, {}
        for (prefix, axis), vals in errs.items():
            m = torch.cat(vals).mean()             # MEAN of per-scene RMSPEs; every axis is
            if axis == 'theta':                    # permutation-matched (the ONE joint
                out[prefix + 'doa_rmspe_deg'] = torch.rad2deg(m)        # permutation)
                norm.setdefault(prefix, {})['theta'] = m / (hi - lo)
            elif axis == 'range':
                out[prefix + 'range_rmspe'] = m
                r_lo, r_hi = min(self.hparams.range_span), max(self.hparams.range_span)
                norm.setdefault(prefix, {})['aux'] = m / max(r_hi - r_lo, 1e-9)
            else:
                out[prefix + 'freq_rmspe_pct'] = 100.0 * m
                norm.setdefault(prefix, {})['aux'] = m / max(2 * self.f_halfspan, 1e-9)
        # co-estimating runs: both axes span-normalized as in the assignment cost, so one
        # number scores the joint task and a checkpoint cannot win on angles while losing
        # the auxiliary axis (only defined where a second axis is active)
        for prefix, axes in norm.items():
            if 'aux' in axes:
                out[prefix + 'joint_rmspe'] = 0.5 * (axes['theta'] + axes['aux'])
        return out

    @torch.no_grad()
    def log_estimation(self, batch):
        for key, val in self.estimation_metrics(batch).items():
            self.log(f'val/{key}', val, on_epoch=True)

    def configure_optimizers(self):
        cls = torch.optim.AdamW if self.hparams.optimizer == 'adamw' else torch.optim.Adam
        return cls(self.parameters(), lr=self.hparams.lr)
