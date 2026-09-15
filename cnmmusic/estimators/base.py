####################################################################################################
#                                             base.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: The estimator base class to inherit from. All machinery is batched-native and fully     #
#          vectorized (batched EVD, einsum spectra, gather-based peak refinement): estimators      #
#          consume (B, M, T) complex snapshot batches and return (B, d) DoAs; a single (M, T)      #
#          sample is treated as a batch of one and returns (d,).                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch

from cnmmusic.arrays.geometry import ArrayGeometry
from cnmmusic.config import DEFAULT_CONFIG
from cnmmusic.arrays.steering import steering_matrix


#**************************************************************************************************#
#                                          Class Estimator                                         #
#**************************************************************************************************#
class Estimator:
    def __init__(self, geom=None, config=None):
        self.default_config = {
            'grid_size': DEFAULT_CONFIG['grid_size'],   # the training scan resolution
            'theta_range': (-1.5533, 1.5533),      # +-89 deg
            'n_src': 2,
            'device': 'cpu',
        }
        if config is not None:
            self.default_config.update({k: v for k, v in config.items()
                                        if k in self.default_config})
        self.__dict__.update(self.default_config)

        self.geom = (geom if geom is not None else ArrayGeometry.ula(8)).to(self.device)
        self.grid = torch.linspace(*self.theta_range, self.grid_size, dtype=torch.float64,
                                   device=self.device)
        self.A_grid = steering_matrix(self.geom.positions[None], self.grid[None])[0]   # (M, G)

    #**********************#
    #   shared machinery   #
    #**********************#
    @staticmethod
    def batched(X):
        """(M, T) -> ((1, M, T), True) or (B, M, T) -> ((B, M, T), False)."""
        X = torch.as_tensor(X)
        return (X[None], True) if X.ndim == 2 else (X, False)

    def covariance(self, X):
        return X @ X.mH / X.shape[-1]

    def noise_subspace(self, R, n_src):
        """Batched EVD; returns En (B, M, M - n_src), ascending eigenvalue order."""
        _, evecs = torch.linalg.eigh(R)
        return evecs[..., :R.shape[-1] - n_src]

    def pick_peaks(self, P, n_src):
        """
        Vectorized peak extraction on spectra P (B, G): top-n_src interior local maxima with
        parabolic (3-point) refinement; falls back to global top-k where fewer maxima exist.
        Returns (B, n_src) sorted DoAs.
        """
        B, G = P.shape
        interior = (P[:, 1:-1] >= P[:, :-2]) & (P[:, 1:-1] >= P[:, 2:])
        vals = torch.where(interior, P[:, 1:-1], torch.full_like(P[:, 1:-1], -torch.inf))
        top_v, top_i = vals.topk(n_src, dim=-1)
        idx = top_i + 1                                            # interior -> grid indices
        fallback = P.topk(n_src, dim=-1).indices.clamp(1, G - 2)   # degenerate spectra
        idx = torch.where(torch.isfinite(top_v), idx, fallback)

        y0 = P.gather(1, idx - 1)
        y1 = P.gather(1, idx)
        y2 = P.gather(1, idx + 1)
        denom = y0 - 2 * y1 + y2
        off = torch.where(denom.abs() > 1e-18, 0.5 * (y0 - y2) / denom,
                          torch.zeros_like(denom))
        doas = self.grid[idx] + off.clamp(-1.0, 1.0) * (self.grid[1] - self.grid[0])
        return doas.sort(dim=-1).values

    def __call__(self, X, n_src=None):
        raise NotImplementedError
