####################################################################################################
#                                             order.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Classical model-order (source-count) estimation from the sample-covariance eigenvalue   #
#          spectrum: AIC and MDL (Wax & Kailath), batched.                                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch


#****************#
#   estimate_d   #
#****************#
def estimate_d(R, T, criterion='aic', d_max=None):
    """
    Wax-Kailath information-criterion source-count estimate from R (B, M, M) with T snapshots.
    Returns d_hat (B,) in 1..d_max (default M - 1).
    """
    ev = torch.linalg.eigvalsh(R).clamp_min(1e-12)                 # ascending, (B, M)
    M = ev.shape[-1]
    d_max = d_max if d_max is not None else M - 1
    costs = []
    for k in range(1, d_max + 1):
        noise = ev[:, :M - k]                                      # smallest M - k eigenvalues
        arith = noise.mean(-1)
        geo = noise.log().mean(-1).exp()
        L = T * (M - k) * torch.log(arith / geo)
        pen = (k * (2 * M - k) if criterion == 'aic'
               else 0.5 * k * (2 * M - k) * torch.log(torch.tensor(float(T))))
        costs.append(L + pen)
    return torch.stack(costs, dim=-1).argmin(-1) + 1
