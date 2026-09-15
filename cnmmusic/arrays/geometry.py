####################################################################################################
#                                           geometry.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Array geometry abstraction and factories (positions in half-wavelength units).          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch

from dataclasses import dataclass, field


#**************************************************************************************************#
#                                          ArrayGeometry                                           #
#**************************************************************************************************#
@dataclass
class ArrayGeometry:
    positions: torch.Tensor          # (M, 3) in half-wavelength units
    name: str = 'array'

    @property
    def M(self):
        return self.positions.shape[-2]

    @property
    def aperture(self):
        """Largest pairwise sensor distance (half-wavelength units)."""
        d = torch.cdist(self.positions[None], self.positions[None])[0]
        return d.max().item()

    def to(self, device=None, dtype=None):
        return ArrayGeometry(self.positions.to(device=device, dtype=dtype), self.name)

    def perturbed(self, delta_pos):
        return ArrayGeometry(self.positions + delta_pos, self.name + '_perturbed')

    #***************#
    #   factories   #
    #***************#
    @classmethod
    def ula(cls, M=8, spacing=1.0):
        pos = torch.zeros(M, 3, dtype=torch.float64)
        pos[:, 0] = spacing * torch.arange(M)
        return cls(pos, 'ula')

    @classmethod
    def nested(cls, M=8, spacing=1.0):
        """Two-level nested array (Pal & Vaidyanathan 2010): N1 = M // 2 elements at the
        base spacing, N2 = M - N1 at (N1 + 1) times it; the difference coarray is a
        contiguous ULA of 2 N2 (N1 + 1) - 1 lags (M = 8 -> 39 lags, virtual ULA of 20)."""
        n1, n2 = M // 2, M - M // 2
        idx = torch.cat([torch.arange(1, n1 + 1, dtype=torch.float64),
                         (n1 + 1) * torch.arange(1, n2 + 1, dtype=torch.float64)]) - 1
        pos = torch.zeros(M, 3, dtype=torch.float64)
        pos[:, 0] = spacing * idx
        return cls(pos, 'nested')

    @classmethod
    def uca(cls, M=8, radius=2.0):
        ang = 2 * torch.pi * torch.arange(M, dtype=torch.float64) / M
        pos = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang),
                           torch.zeros_like(ang)], dim=-1)
        return cls(pos, 'uca')

    @classmethod
    def ura(cls, Mx=4, My=2, spacing=1.0):
        gx, gy = torch.meshgrid(torch.arange(Mx, dtype=torch.float64),
                                torch.arange(My, dtype=torch.float64), indexing='ij')
        pos = torch.stack([spacing * gx.flatten(), spacing * gy.flatten(),
                           torch.zeros(Mx * My, dtype=torch.float64)], dim=-1)
        return cls(pos, 'ura')

    @classmethod
    def nonuniform_la(cls, positions_1d):
        p = torch.as_tensor(positions_1d, dtype=torch.float64)
        pos = torch.zeros(len(p), 3, dtype=torch.float64)
        pos[:, 0] = p
        return cls(pos, 'nula')

    @classmethod
    def random_planar(cls, M=8, aperture=7.0, min_spacing=0.5, rng=None, max_tries=1000):
        """Random 2-D array with a minimum sensor spacing, rescaled to the target aperture."""
        rng = rng if rng is not None else torch.Generator().manual_seed(0)
        pts = []
        for _ in range(max_tries):
            cand = aperture * (torch.rand(2, generator=rng, dtype=torch.float64) - 0.5)
            if all((cand - p).norm() >= min_spacing for p in pts):
                pts.append(cand)
            if len(pts) == M:
                break
        if len(pts) < M:
            raise RuntimeError(f'could not place {M} sensors with min spacing {min_spacing}')
        xy = torch.stack(pts)
        xy = xy - xy.mean(0, keepdim=True)
        d = torch.cdist(xy[None], xy[None])[0]
        xy = xy * (aperture / d.max())
        pos = torch.cat([xy, torch.zeros(M, 1, dtype=torch.float64)], dim=-1)
        return cls(pos, 'random_planar')


#**************************************************************************************************#
#                                         GeometrySampler                                          #
#**************************************************************************************************#
class GeometrySampler:
    """Samples array geometries for geometry-randomized training."""

    def __init__(self, kinds=('ula',), M_range=(8, 8), aperture_range=(7.0, 7.0), seed=0):
        self.kinds = kinds
        self.M_range = M_range
        self.aperture_range = aperture_range
        self.rng = torch.Generator().manual_seed(seed)

    def sample(self):
        kind = self.kinds[torch.randint(len(self.kinds), (1,), generator=self.rng).item()]
        M = torch.randint(self.M_range[0], self.M_range[1] + 1, (1,), generator=self.rng).item()
        lo, hi = self.aperture_range
        aperture = lo + (hi - lo) * torch.rand(1, generator=self.rng, dtype=torch.float64).item()
        if kind == 'ula':
            return ArrayGeometry.ula(M, spacing=aperture / max(M - 1, 1))
        if kind == 'uca':
            return ArrayGeometry.uca(M, radius=aperture / 2)
        if kind == 'ura':
            Mx = max(2, int(M ** 0.5))
            return ArrayGeometry.ura(Mx, max(M // Mx, 1), spacing=aperture / max(Mx - 1, 1))
        if kind == 'nula':
            gaps = 0.5 + torch.rand(M - 1, generator=self.rng, dtype=torch.float64)
            p = torch.cat([torch.zeros(1, dtype=torch.float64), gaps.cumsum(0)])
            return ArrayGeometry.nonuniform_la(p * aperture / p[-1])
        if kind == 'random_planar':
            return ArrayGeometry.random_planar(M, aperture, rng=self.rng)
        raise ValueError(f'unknown geometry kind: {kind}')
