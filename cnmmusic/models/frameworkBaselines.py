####################################################################################################
#                                      frameworkBaselines.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Lightning framework training the learned baselines (SubspaceNet through the same        #
#          differentiable Root-MUSIC layer as CNM, DA-MUSIC end-to-end regression, GridCNN grid    #
#          classification) on identical simulator batches.                                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch

from cnmmusic.arrays.steering import steering_matrix
from cnmmusic.models.framework import Framework, condition_groups, sub_batch
from cnmmusic.archs.baselines import SubspaceNet, DAMUSIC, GridCNN
from cnmmusic.archs.music import DifferentiableRootMUSIC
from cnmmusic.criteria.losses import doa_loss
from cnmmusic.criteria.metrics import rmspe


#**************************************************************************************************#
#                                    Class BaselinesFramework                                      #
#**************************************************************************************************#
class BaselinesFramework(Framework):
    def __init__(self, baseline='subspacenet', M=8, n_src=2, snapshots=100, tau=8,
                 grid_size=121, theta_range=(-1.05, 1.05), lr=1e-3, loss='rmspe',
                 weight_decay=0.0, train_cov='empirical', **kwargs):
        super().__init__(baseline=baseline, M=M, n_src=n_src, snapshots=snapshots, tau=tau,
                         grid_size=grid_size, theta_range=tuple(theta_range), lr=lr, loss=loss,
                         weight_decay=weight_decay, train_cov=train_cov)
        if baseline == 'subspacenet':
            self.net = SubspaceNet(N=M, tau=tau)
            self.root = DifferentiableRootMUSIC()
        elif baseline == 'damusic':                                 # any-T, variable-d variant,
            self.net = DAMUSIC(N=M, grid_size=grid_size,            # same scan grid as MUSIC/CNM
                               theta_range=theta_range)
        elif baseline == 'gridcnn':
            self.net = GridCNN(N=M, grid_size=grid_size, theta_range=theta_range)
        else:
            raise ValueError(f'unknown baseline: {baseline}')
        self._A_pos, self._A = None, None       # damusic: manifold cache keyed on positions

    #*************#
    #   forward   #
    #*************#
    def _manifold(self, positions):
        """Aperture steering vectors for damusic, rebuilt only when the positions change."""
        if self._A_pos is None or not torch.equal(self._A_pos, positions):
            self._A_pos = positions.clone()
            self._A = steering_matrix(positions[None].to(torch.float64),
                                      self.net.grid[None].to(positions.device))[0]
        return self._A

    def forward(self, X, n_src, positions=None, R=None):
        if self.hparams.baseline == 'subspacenet':
            Rz = self.net(X)
            jitter = 1e-9 * torch.eye(Rz.shape[-1], dtype=Rz.dtype, device=Rz.device)
            _, evecs = torch.linalg.eigh(Rz + jitter)
            En = evecs[..., :Rz.shape[-1] - n_src]
            return self.root(En, n_src)
        if self.hparams.baseline == 'damusic':                      # known d -> hard subspace;
            return self.net(X, n_src, self._manifold(positions))    # aperture = batch data
        return self.net(X, R)                                       # gridcnn: logits

    #*************************#
    #   lightning interface   #
    #*************************#
    def step(self, batch):
        """Batches mix T and d per element: one forward per dense (T, d) group."""
        B = batch['X'].shape[0]
        total, errs = 0.0, []
        for idx, T, d in condition_groups(batch):
            sub = sub_batch(batch, idx, T, d)
            w = len(idx) / B
            X = sub['X'][:, 0]
            doas = sub['doas'][:, 0]
            # gridcnn trains on the exact covariance when train_cov='true' (the reference
            # protocol) and always validates on the sample one
            R = (sub['R_true'][:, 0] if self.hparams.train_cov == 'true' and self.training
                 else None)
            out = self.forward(X, d, sub['positions'][0], R)
            if self.hparams.baseline == 'gridcnn':
                loss_g = torch.nn.functional.binary_cross_entropy_with_logits(
                    out, self.net.targets(doas))
                pred = self.net.decode(out, d)
            elif self.hparams.baseline == 'damusic':
                pred = out[:, :d]                       # known d: first-d slice
                loss_g = doa_loss(pred, doas, root=self.hparams.loss == 'rmspe')
            else:
                loss_g = doa_loss(out, doas, root=self.hparams.loss == 'rmspe')
                pred = out
            total = total + w * loss_g
            errs.append(rmspe(pred.detach(), doas, reduce=False))
        return total, torch.rad2deg(torch.cat(errs).mean())   # MEAN of per-scene RMSPE,
                                                              # as frameworkCNM logs it

    def training_step(self, batch, batch_idx):
        loss, err = self.step(batch)
        self.log_dict({'train/loss': loss, 'train/rmspe_deg': err}, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, err = self.step(batch)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val/rmspe_deg', err, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
