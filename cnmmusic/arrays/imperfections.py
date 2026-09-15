####################################################################################################
#                                         imperfections.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Array imperfection models generating ground-truth perturbed manifolds.                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch

from dataclasses import dataclass

from cnmmusic.arrays.steering import steering_matrix, nearfield_steering_matrix, gauge_fix


#**************************************************************************************************#
#                                       ImperfectionParams                                         #
#**************************************************************************************************#
@dataclass
class ImperfectionParams:
    delta_pos: torch.Tensor       # (M, 3) position perturbations, row 0 = 0
    gain: torch.Tensor            # (M,)   multiplicative gains, nominal 1
    phase: torch.Tensor           # (M,)   phase offsets [rad], entry 0 = 0
    coupling: torch.Tensor        # (M, M) complex coupling matrix, nominal I
    rho: float = 1.0

    def flatten(self):
        """Real vector of all parameters (for logging / z-supervision targets)."""
        return torch.cat([self.delta_pos.flatten(), self.gain, self.phase,
                          self.coupling.real.flatten(), self.coupling.imag.flatten()])

    @staticmethod
    def identity(M):
        return ImperfectionParams(torch.zeros(M, 3, dtype=torch.float64),
                                  torch.ones(M, dtype=torch.float64),
                                  torch.zeros(M, dtype=torch.float64),
                                  torch.eye(M, dtype=torch.complex128), rho=0.0)


#**************************************************************************************************#
#                                        ImperfectionModel                                         #
#**************************************************************************************************#
class ImperfectionModel:
    """
    Samples ImperfectionParams and builds ground-truth perturbed manifolds.

    kinds: subset of ('position', 'gain', 'phase', 'coupling'); rho in [0, inf) scales all maxima.
    randomized: per-draw U(-max, max) errors (training); False gives the deterministic canonical
    Liu-2018 pattern (half the sensors +max, half -max) for reproducible benchmarks.
    """

    def __init__(self, kinds=('position', 'gain', 'phase', 'coupling'), rho=1.0, randomized=True,
                 max_pos=0.2, max_gain=0.2, max_phase_deg=30.0,
                 coupling_mag=0.3, coupling_phase_deg=60.0, coupling_bandwidth=None, seed=0,
                 bounds=None):
        self.kinds = kinds
        self.rho = rho
        self.bounds = bounds       # dict of (lo, hi) uniform bounds per type; overrides rho
        self.randomized = randomized
        self.max_pos = max_pos
        self.max_gain = max_gain
        self.max_phase = torch.deg2rad(torch.tensor(max_phase_deg)).item()
        self.gamma = coupling_mag * torch.exp(1j * torch.deg2rad(torch.tensor(coupling_phase_deg,
                                                                              dtype=torch.float64)))
        self.coupling_bandwidth = coupling_bandwidth
        self.rng = torch.Generator().manual_seed(seed)

    #**************#
    #   sampling   #
    #**************#
    def _draw(self, M, max_val):
        if self.randomized:
            return max_val * (2 * torch.rand(M, generator=self.rng, dtype=torch.float64) - 1)
        signs = torch.ones(M, dtype=torch.float64)                  # canonical: half +, half -
        signs[:M // 2] = -1.0
        return max_val * signs

    def _bound(self, key):
        """Per-condition severity scale for one imperfection type from its uniform bounds."""
        lo, hi = self.bounds[key]
        return lo + (hi - lo) * torch.rand(1, generator=self.rng, dtype=torch.float64).item()

    def sample(self, M, rho=None):
        params = ImperfectionParams.identity(M)
        if self.bounds is not None:                                 # bounds mode (training)
            s_pos = self._bound('pos') if 'pos' in self.bounds else 0.0
            s_gain = self._bound('gain') if 'gain' in self.bounds else 0.0
            s_phase = torch.deg2rad(torch.tensor(
                self._bound('phase_deg') if 'phase_deg' in self.bounds else 0.0)).item()
            c_mag = self._bound('coupling_mag') if 'coupling_mag' in self.bounds else 0.0
            c_ph = torch.deg2rad(torch.tensor(
                self._bound('coupling_phase_deg') if 'coupling_phase_deg' in self.bounds
                else 0.0)).item()
            gamma = c_mag * torch.exp(1j * torch.tensor(c_ph, dtype=torch.float64))
        else:                                                       # rho mode (evaluation sweeps)
            rho = self.rho if rho is None else rho
            params.rho = rho
            s_pos, s_gain = rho * self.max_pos, rho * self.max_gain
            s_phase, gamma = rho * self.max_phase, rho * self.gamma

        if 'position' in self.kinds:
            dx = self._draw(M, s_pos)
            dx[0] = 0.0                                             # gauge reference sensor
            params.delta_pos = torch.zeros(M, 3, dtype=torch.float64)
            params.delta_pos[:, 0] = dx
        if 'gain' in self.kinds:
            g = 1.0 + self._draw(M, s_gain)
            g[0] = 1.0
            params.gain = g
        if 'phase' in self.kinds:
            p = self._draw(M, s_phase)
            p[0] = 0.0
            params.phase = p
        if 'coupling' in self.kinds:
            k = torch.arange(M, dtype=torch.float64)
            prof = gamma ** k                                       # gamma^|i-j| Toeplitz profile
            prof[0] = 1.0
            if self.coupling_bandwidth is not None:
                prof[self.coupling_bandwidth + 1:] = 0.0
            idx = (k[:, None] - k[None, :]).abs().long()
            params.coupling = prof[idx]
        return params

    #***************************#
    #   ground-truth manifold   #
    #***************************#
    def true_manifold(self, geom, params, theta, r=None, f=1.0, gauge=True):
        """
        Ground-truth perturbed steering matrix a_true(theta[, r]) of shape (M, G).
        """
        pos = (geom.positions + params.delta_pos)[None]             # (1, M, 3)
        if r is None:
            a0 = steering_matrix(pos, torch.as_tensor(theta)[None], f)[0]
        else:
            a0 = nearfield_steering_matrix(pos, torch.as_tensor(theta)[None],
                                           torch.as_tensor(r)[None], f)[0]
        d = (params.gain * torch.exp(1j * params.phase)).to(a0.dtype)
        a = params.coupling.to(a0.dtype) @ (d[:, None] * a0)
        return gauge_fix(a) if gauge else a
