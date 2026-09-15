####################################################################################################
#                                           beamform.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Beamforming with (learned) steering vectors: Bartlett and MVDR weights plus output-     #
#          SINR evaluation against the true array response — the mismatch-sensitivity testbed.     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch


#*************#
#   weights   #
#*************#
def bartlett_weights(a):
    """w = a / M for steering vectors a (..., M)."""
    return a / a.shape[-1]


def mvdr_weights(R, a, loading=1e-4):
    """
    Capon weights w = R^-1 a / (a^H R^-1 a) with relative diagonal loading, batched.
    R (..., M, M), a (..., M) -> w (..., M).
    """
    M = R.shape[-1]
    load = loading * R.diagonal(dim1=-2, dim2=-1).real.mean(-1)[..., None, None]
    Rinv_a = torch.linalg.solve(R + load * torch.eye(M, dtype=R.dtype, device=R.device),
                                a[..., None])[..., 0]
    denom = (a.conj() * Rinv_a).sum(-1, keepdim=True)
    return Rinv_a / denom


#*****************#
#   output sinr   #
#*****************#
def output_sinr(w, a_true_tgt, p_tgt, a_true_int=None, p_int=None, sigma2=1.0):
    """
    SINR at the beamformer output given TRUE array responses: w, a_true_* (..., M);
    p_tgt/p_int source powers; sigma2 noise power. Returns linear SINR (...,).
    """
    sig = p_tgt * (w.conj() * a_true_tgt).sum(-1).abs().pow(2)
    noise = sigma2 * w.abs().pow(2).sum(-1)
    inter = (p_int * (w.conj() * a_true_int).sum(-1).abs().pow(2)
             if a_true_int is not None else 0.0)
    return sig / (noise + inter)
