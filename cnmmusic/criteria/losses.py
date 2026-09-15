####################################################################################################
#                                            losses.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Training losses for the conditional steering-manifold generator: gauge-invariant        #
#          manifold alignment, direction-discriminative ranking on the null spectrum, covariance   #
#          reconstruction, angular smoothness, and the permutation-matched DoA loss (all           #
#          differentiable; the Hungarian assignment itself is computed on detached costs).         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch

from scipy.optimize import linear_sum_assignment


#************************#
#   manifold alignment   #
#************************#
def alignment_loss(a_hat, a_true, dim=-2):
    """
    Gauge-invariant subspace alignment: 1 - |a_true^H a_hat|^2 / (||a_true||^2 ||a_hat||^2),
    averaged over all remaining dims. a_hat, a_true: (..., M, G).
    """
    num = (a_true.conj() * a_hat).sum(dim).abs().pow(2)
    den = a_true.abs().pow(2).sum(dim) * a_hat.abs().pow(2).sum(dim)
    return (1.0 - num / den.clamp_min(1e-12)).mean()


#*********************************#
#   normalized subspace leakage   #
#*********************************#
def subspace_ratio(En, a):
    """
    J = ||En^H a||^2 / ||a||^2 per steering vector: the fraction of a's energy inside the noise
    subspace, in [0, 1] for orthonormal En (a projector ratio -- scale- and gauge-invariant, so
    the trivial fix of shrinking ||a|| buys nothing). En (..., M, K), a (..., M, n) -> (..., n).
    """
    num = torch.einsum('...mk,...mn->...kn', En.conj(), a).abs().pow(2).sum(-2)
    den = a.abs().pow(2).sum(-2).clamp_min(1e-30)
    return num / den


#******************************#
#   ranking (discriminative)   #
#******************************#
def ranking_loss(q_true, q_neg, margin=2.0):
    """
    Hinge on log null-spectra: the true directions' q must undercut negatives by the margin.
    q_true: (..., d), q_neg: (..., n_neg) -> softplus(margin + log q_true - log q_neg).
    """
    lt = torch.log(q_true.clamp_min(1e-30))[..., :, None]
    ln = torch.log(q_neg.clamp_min(1e-30))[..., None, :]
    return torch.nn.functional.softplus(margin + lt - ln).mean()


#**************************#
#   landscape likelihood   #
#**************************#
def likelihood_loss(q_true, q_neg, tau=1.0, excl_mask=None):
    """
    The null landscape read as an unnormalized density p(tuple) ~ exp(-log J / tau): cross-
    entropy of each truth under the softmax over {itself, the sampled negatives} (the negatives
    Monte-Carlo the partition function). Saturates naturally once the truth wins (gradient
    (1 - p_true) / tau -> 0, no margin to chase forever) and concentrates gradient on the
    DEEPEST spurious null automatically. excl_mask (True = inside a truth's resolution cell)
    removes candidates that are not rival hypotheses. q_true (..., d), q_neg (..., n).
    """
    lt = -torch.log(q_true.clamp_min(1e-30))[..., :, None] / tau           # (..., d, 1)
    ln = -torch.log(q_neg.clamp_min(1e-30)) / tau                          # (..., n)
    if excl_mask is not None:
        ln = ln.masked_fill(excl_mask, -torch.inf)
    ln = ln[..., None, :].expand(*lt.shape[:-1], ln.shape[-1])             # (..., d, n)
    logits = torch.cat([lt, ln], dim=-1)
    return -torch.log_softmax(logits, dim=-1)[..., 0].mean()


#*********************************#
#   landscape kl (soft targets)   #
#*********************************#
def landscape_kl_loss(q_neg, q_target, tau=1.0):
    """
    KL between the physics target density Q(p) ~ sum_i exp(-deficit_i(p) / eps) (the ideal
    pseudospectrum: one bump per truth, width = the physical resolution) and the model
    landscape density P(p) ~ J(p)^(-1/tau), both estimated on the same uniform candidate
    sample by self-normalized importance weights -- invariant to the candidate count, which
    only refines the estimate. Truth depth is NOT enforced here (that is the orthogonality
    term); this term only shapes WHERE the landscape's mass goes.
    q_neg (..., n) model leakages at the candidates, q_target (..., n) unnormalized Q values.
    """
    logp = -torch.log(q_neg.clamp_min(1e-30)) / tau
    logp = logp - torch.logsumexp(logp, dim=-1, keepdim=True)              # log p-bar
    qb = q_target / q_target.sum(-1, keepdim=True).clamp_min(1e-30)       # q-bar
    logq = torch.log(qb.clamp_min(1e-30))
    kl = qb * (logq - logp)
    return torch.where(qb > 0, kl, torch.zeros_like(kl)).sum(-1).mean()


#**************************************#
#   beam-target landscape likelihood   #
#**************************************#
def likelihood_loss_beam(q_true, q_neg, overlap, tau=1.0):
    """
    Cross-entropy between the model's landscape softmax p ~ exp(-log J / tau) and the array's
    OWN beampattern: target mass [1, overlap_1, ..., overlap_n] per truth, where overlap is the
    normalized nominal-steering coherence |a0(truth)^H a0(cand)|^2 / (||.||^2 ||.||^2). The soft
    ball around each truth is exactly the main lobe -- per geometry, frequency, and range, with
    no radius, temperature, or mask parameters. q_true (..., d), q_neg (..., n),
    overlap (..., d, n) in [0, 1].
    """
    lt = -torch.log(q_true.clamp_min(1e-30))[..., :, None] / tau           # (..., d, 1)
    ln = (-torch.log(q_neg.clamp_min(1e-30)) / tau)[..., None, :]          # (..., 1, n)
    ln = ln.expand(*lt.shape[:-1], ln.shape[-1])
    logits = torch.cat([lt, ln], dim=-1)                                   # (..., d, 1 + n)
    targets = torch.cat([torch.ones_like(lt), overlap], dim=-1)
    targets = targets / targets.sum(-1, keepdim=True)
    return -(targets * torch.log_softmax(logits, dim=-1)).sum(-1).mean()


#*******************************#
#   covariance reconstruction   #
#*******************************#
def cov_recon_loss(R_hat, A, R_s, sigma2):
    """
    || R_hat - A R_s A^H - sigma2 I ||_F^2 / || R_hat ||_F^2 with A (..., M, d) at the true DoAs.
    """
    M = R_hat.shape[-1]
    R_model = A @ R_s.to(A.dtype) @ A.mH
    eye = torch.eye(M, dtype=A.dtype, device=A.device)
    R_model = R_model + sigma2[..., None, None].to(A.dtype) * eye
    num = (R_hat - R_model).abs().pow(2).sum((-2, -1))
    den = R_hat.abs().pow(2).sum((-2, -1)).clamp_min(1e-12)
    return (num / den).mean()


#************************#
#   angular smoothness   #
#************************#
def smoothness_loss(a_hat, a_nom):
    """
    Mean squared second difference of the CORRECTION ratio a_hat / a_nom along the angle grid
    (the nominal manifold's intrinsic curvature is not penalized; exactly zero at nominal).
    a_hat, a_nom: (..., M, G) on consecutive grid angles.
    """
    ratio = a_hat / (a_nom + 1e-12)
    d2 = ratio[..., 2:] - 2 * ratio[..., 1:-1] + ratio[..., :-2]
    return d2.abs().pow(2).mean()


#*****************************#
#   permutation-matched doa   #
#*****************************#
def doa_loss(pred, true, period=2 * torch.pi, root=False):
    """
    Wrapped squared error after Hungarian matching (assignment on detached costs; gradients flow
    through the matched predictions). root=True takes the per-sample square root (RMSPE, the
    DA-MUSIC / SubspaceNet reference objective). pred, true: (B, d).
    """
    diff = (pred[..., :, None] - true[..., None, :]) % period
    diff = torch.where(diff > period / 2, diff - period, diff)
    cost = diff.pow(2)
    losses = []
    for b in range(cost.shape[0]):
        ri, ci = linear_sum_assignment(cost[b].detach().cpu().numpy())
        sample = cost[b, ri, ci].mean()
        losses.append(sample.clamp_min(1e-12).sqrt() if root else sample)
    return torch.stack(losses).mean()
