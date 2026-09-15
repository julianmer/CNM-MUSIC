####################################################################################################
#                                            metrics.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Evaluation metrics (permutation-invariant wrapped RMSPE, resolution probability).       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch

from itertools import permutations
from scipy.optimize import linear_sum_assignment


#********************#
#   angular errors   #
#********************#
@torch.no_grad()
def wrapped_diff(a, b, period=2 * torch.pi):
    """Wrapped angular difference in (-period/2, period/2]."""
    d = (a - b) % period
    return torch.where(d > period / 2, d - period, d)


@torch.no_grad()
def rmspe(pred, true, period=2 * torch.pi, reduce=True):
    """
    Permutation-invariant wrapped RMSE between DoA sets (d,) or batches (B, d).
    Uses Hungarian assignment on the wrapped squared-error cost (exact, scales past d! limits).
    reduce=False returns the per-scene RMSPE values (B,) instead of the batch RMS.
    """
    if pred.ndim == 1:
        pred, true = pred[None], true[None]
    errs = []
    for p, t in zip(pred, true):
        cost = wrapped_diff(p[:, None], t[None, :], period).pow(2)
        ri, ci = linear_sum_assignment(cost.cpu().numpy())
        errs.append(cost[ri, ci].mean())
    errs = torch.stack(errs)
    return errs.mean().sqrt() if reduce else errs.sqrt()


@torch.no_grad()
def matched_tuple_rmse(pred, true, aux=(), theta_span=2 * torch.pi, period=2 * torch.pi):
    """
    ONE minimal permutation per scene over whole sources: the Hungarian
    cost is the sum of SPAN-NORMALIZED squared differences of every axis present -- wrapped
    angle over theta_span plus each auxiliary axis over its span -- so no axis dominates the
    assignment, and every co-estimated quantity inherits the same permutation.

    pred/true (B, d) angles. aux: iterable of (pred_aux, true_aux, span[, rep_pred, rep_true])
    tuples; the cost uses (diff/span)^2, the returned RMSEs use the report values (default:
    the cost values). Returns (theta_rmspe (B,), [aux_rmse (B,), ...]) in native units.
    """
    if pred.ndim == 1:
        pred, true = pred[None], true[None]
        aux = [(a[0][None], a[1][None], a[2], *[x[None] for x in a[3:]]) for a in aux]
    th_errs, aux_errs = [], [[] for _ in aux]
    for b in range(pred.shape[0]):
        cost = (wrapped_diff(pred[b][:, None], true[b][None, :], period) / theta_span).pow(2)
        for pa, ta, span, *_ in aux:
            cost = cost + ((pa[b][:, None] - ta[b][None, :]) / span).pow(2)
        ri, ci = linear_sum_assignment(cost.cpu().numpy())
        th_errs.append(wrapped_diff(pred[b][ri], true[b][ci], period).pow(2).mean().sqrt())
        for k, a in enumerate(aux):
            rp, rt = (a[3], a[4]) if len(a) > 3 else (a[0], a[1])
            aux_errs[k].append((rp[b][ri] - rt[b][ci]).pow(2).mean().sqrt())
    return torch.stack(th_errs), [torch.stack(e) for e in aux_errs]


@torch.no_grad()
def prob_resolution(pred, true, tol=None, period=2 * torch.pi):
    """
    Probability of successful resolution for a batch of two-source cases: both sources matched
    within tol (default: half the true separation).
    """
    ok = []
    for p, t in zip(pred, true):
        cost = wrapped_diff(p[:, None], t[None, :], period).abs()
        ri, ci = linear_sum_assignment(cost.cpu().numpy())
        lim = tol if tol is not None else wrapped_diff(t[0], t[1], period).abs() / 2
        ok.append((cost[ri, ci] < lim).all())
    return torch.stack(ok).double().mean()


#********************************#
#   unknown-source-count rmspe   #
#********************************#
@torch.no_grad()
def rmspe_est_d(pred_pad, d_hat, true, period=2 * torch.pi):
    """
    Error under estimated source counts, following the DA-MUSIC convention: an undercount is
    padded with random angles, an overcount drops random predictions, then the full
    d_true x d_true permutation RMSPE is scored. Returns (rmspe, count accuracy).
    pred_pad (B, d_max) NaN-padded, d_hat (B,), true (B, d_true).
    """
    errs, hits = [], []
    d_true = true.shape[-1]
    for b in range(true.shape[0]):
        k = int(d_hat[b].item())
        hits.append(k == d_true)
        preds = pred_pad[b, :k]
        if k < d_true:                                             # too few: add random angles
            extra = (torch.rand(d_true - k, dtype=true.dtype) - 0.5) * period
            preds = torch.cat([preds, extra])
        elif k > d_true:                                           # too many: drop random ones
            preds = preds[torch.randperm(k)[:d_true]]
        cost = wrapped_diff(preds[:, None], true[b][None, :], period).pow(2)
        ri, ci = linear_sum_assignment(cost.cpu().numpy())
        errs.append(cost[ri, ci].mean())
    return torch.stack(errs).mean().sqrt(), torch.tensor(hits, dtype=torch.float64).mean()
