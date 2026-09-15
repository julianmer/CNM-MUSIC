####################################################################################################
#                                            layers.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Building-block layers (SIREN sine layers, FiLM conditioning, Fourier features).         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch
import torch.nn as nn


#**************************************************************************************************#
#                                          Class SineLayer                                         #
#**************************************************************************************************#
#                                                                                                  #
# SIREN layer y = sin(omega0 * (W x + b)) with the initialization of Sitzmann et al. (2020).       #
#                                                                                                  #
#**************************************************************************************************#
class SineLayer(nn.Module):
    def __init__(self, in_dim, out_dim, omega0=30.0, first=False):
        super().__init__()
        self.omega0 = omega0
        self.linear = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            bound = 1.0 / in_dim if first else (6.0 / in_dim) ** 0.5 / omega0
            self.linear.weight.uniform_(-bound, bound)
        self.first = first

    def forward(self, x):
        return torch.sin(self.omega0 * self.linear(x))


#**************************************************************************************************#
#                                          Class FiLMLayer                                         #
#**************************************************************************************************#
#                                                                                                  #
# Feature-wise linear modulation: y = gamma(z) * x + beta(z), with gamma initialized at 1 and      #
# beta at 0 so an untrained conditioning path is the identity.                                     #
#                                                                                                  #
#**************************************************************************************************#
class FiLMLayer(nn.Module):
    def __init__(self, z_dim, feat_dim):
        super().__init__()
        self.gamma = nn.Linear(z_dim, feat_dim)
        self.beta = nn.Linear(z_dim, feat_dim)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)

    def forward(self, x, z):
        # z: (B, z_dim) modulating x: (B, ..., feat_dim)
        expand = (slice(None),) + (None,) * (x.ndim - 2)
        return (1.0 + self.gamma(z)[expand]) * x + self.beta(z)[expand]


#**********************#
#   fourier features   #
#**********************#
def fourier_features(theta, n=8):
    """Multi-scale angle features [sin(2^k theta), cos(2^k theta)] for k = 0..n-1 -> (..., 2n)."""
    scales = 2.0 ** torch.arange(n, dtype=theta.dtype, device=theta.device)
    ang = theta[..., None] * scales
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


#**********************#
#   zero-init linear   #
#**********************#
def zero_linear(in_dim, out_dim):
    """Linear layer initialized to output exactly zero (for correction heads that start at nominal)."""
    lin = nn.Linear(in_dim, out_dim)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin
