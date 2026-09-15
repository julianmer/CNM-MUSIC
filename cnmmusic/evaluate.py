####################################################################################################
#                                           evaluate.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Monte-Carlo benchmark harness: sweep one scenario parameter over a grid, run every      #
#          estimator on identical realizations, cache per-(scenario, method) RMSPE results to      #
#          HDF5 (resume-aware), and plot the resulting curves.                                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import h5py
import math
import time
import pathlib
import torch

from tqdm import tqdm

from cnmmusic.data.simulator import NarrowbandSimulator
from cnmmusic.criteria.metrics import rmspe


#*******************#
#   runexperiment   #
#*******************#
def runExperiment(name, param, values, estimators, base_config=None, n_mc=1000, mc_batch=500,
                  cache_dir='results', seed=1234, oracle_key='oracle', device='cpu',
                  known_n_src=True):
    """
    Sweeps `param` over `values`; for each value draws n_mc Monte-Carlo conditions and runs every
    estimator in `estimators` (dict name -> callable) on identical data, one BATCHED call per
    estimator per MC batch (estimators are batched-native). Estimators named `oracle_key`
    additionally receive the ground-truth manifold. Results are cached to {cache_dir}/{name}.h5
    per (value, method): mean RMSPE over all scenes for estimators and the Bayesian
    (Van Trees) CRB -- averaged Fisher information, the bound on that mean -- as the
    reference; finished entries are skipped on rerun.

    known_n_src=False: the true source count is withheld -- estimators with native count estimation
    (attribute native_d) use their own d_hat; all others use AIC on the sample covariance.
    Additional '{method}_dacc' entries record count accuracy; CRB always uses the true d.
    """
    cache = pathlib.Path(cache_dir) / f'{name}.h5'
    cache.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(cache, 'a') as h5:
        for val in values:
            cfg = dict(base_config or {})
            cfg['seed'] = seed          # common random numbers: the SAME scenes at every sweep
                                        # point, so a curve tracks the axis, not the draw
            if param in ('snr', 'snapshots', 'n_src'):                 # scalar -> range params
                cfg[f'{param}_range'] = (val, val)
            elif param == 'range':                                     # all sources at distance val
                cfg['range_range'] = (val, val)
            elif param == 'delta':                                     # exact 2-source separation [deg]
                cfg['n_src_range'] = (2, 2)
                cfg['fixed_sep'] = math.radians(val)
            elif param == 'freq':                                  # % off the design carrier
                f0 = 1.0 + val / 100.0
                cfg['freq_range'] = (f0, f0)
            elif param == 'f_center':       # per-source carriers in the +-0.05 bin around val
                lo, hi = cfg.get('freq_range', (0.0, 1.0))     # (distinct carriers: identical
                cfg['freq_range'] = (max(val - 0.05, lo), min(val + 0.05, hi))  # tones are coherent)
            elif param == 'spread_deg':                                # every source spread val
                cfg['spread_deg'] = (val, val)
            elif param == 'n_failed':                                  # exactly val dead sensors
                cfg['n_failed'] = (val, val)
            elif param == 'source_corr':
                cfg['source_corr'] = val
                cfg['source_corr_range'] = None   # pin: a marginal draw would override the scalar
            elif param == 'noise_corr':                                # AR-1 noise coefficient
                cfg['noise_corr_range'] = (val, val)
            elif param == 'rho':
                from cnmmusic.arrays.imperfections import ImperfectionModel
                cfg['imperfections'] = ImperfectionModel(rho=val, randomized=True,
                                                         seed=seed)
                cfg['rho_range'] = None            # pin: a marginal draw would override the model
            else:
                cfg[param] = val
            sim = NarrowbandSimulator(cfg)

            todo = {n: e for n, e in estimators.items() if f'{param}={val}/{n}' not in h5}
            if not todo:
                continue
            errs = {n: [] for n in todo}
            secs = {n: [] for n in todo}      # per-scene inference time, free of extra passes
            for _ in tqdm(range(max(n_mc // mc_batch, 1)), desc=f'{param}={val}', leave=False):
                b = sim.sample(mc_batch)                    # one draw scored by every estimator
                X = b['X'][:, 0].to(device)
                doas = b['doas'][:, 0]
                if not known_n_src:
                    from cnmmusic.estimators.order import estimate_d
                    d_aic = estimate_d(X @ X.mH / X.shape[-1], X.shape[-1])
                for est_name, est in todo.items():
                    if est_name == 'CRB':
                        from cnmmusic.utils.crb import crb_from_batch, nearfield_crb, freq_crb
                        # per-scene bounds are clipped at the uniform-prior variance (W^2/12):
                        # beyond "random guessing over the prior" the local bound is vacuous
                        # and its heavy tail would dominate every scene average
                        W = cfg['theta_range'][1] - cfg['theta_range'][0]
                        far = torch.isinf(b['ranges']).all()
                        if param == 'rho' and far:
                            # calibration errors are unknown to every estimator, so the
                            # reference is the HYBRID bound (prior on the perturbations),
                            # not the genie bound that knows the perturbed manifold
                            from cnmmusic.utils.crb import hybrid_crb_from_batch
                            errs[est_name].append(
                                hybrid_crb_from_batch(b, b['positions'][0], float(val),
                                                      prior_var=W ** 2 / 12, reduce=False))
                        else:
                            errs[est_name].append(crb_from_batch(b, b['positions'][0],
                                                                 prior_var=W ** 2 / 12,
                                                                 reduce=False))
                        r_true = b['ranges'][:, 0]
                        if torch.isfinite(r_true).all():           # joint (theta, r) bound
                            _, vr = nearfield_crb(b['positions'][0], doas, r_true,
                                                  b['R_s'][:, 0], b['sigma2'], b['X'].shape[-1])
                            if cfg.get('range_range'):
                                rw = cfg['range_range'][1] - cfg['range_range'][0]
                                vr = vr.clamp(max=rw ** 2 / 12)
                            errs.setdefault('CRB_range', []).append(vr.mean(-1).sqrt())
                        if bool((b['freqs'] != 1.0).any()):        # joint (theta, f) bound [% f_c]
                            _, vf = freq_crb(b['positions'][0], doas, b['freqs'],
                                             b['R_s'][:, 0], b['sigma2'], b['X'].shape[-1])
                            vf = vf.clamp(max=0.5 ** 2 / 12)       # F-MUSIC scan width as prior
                            errs.setdefault('CRB_freq', []).append(100.0 * vf.mean(-1).sqrt())
                        continue
                    if est_name == 'ZZB':
                        from cnmmusic.utils.zzb import zzb_from_batch
                        errs[est_name].append(zzb_from_batch(b, b['positions'][0],
                                                             cfg['theta_range'], n_scenes=8))
                        continue
                    kwargs = {}
                    if est_name == oracle_key:
                        kwargs['A_true'] = b['a_true_grid'].to(device)
                    elif est_name.startswith('oracle-root'):
                        kwargs['imperfect'] = b['imperfect'].to(device)
                    if known_n_src:
                        # sweep batches pin every axis: one (T, d) for the whole batch
                        _t0 = time.perf_counter()
                        preds = est(X, n_src=int(b['n_src'][0]), **kwargs)
                        secs[est_name].append((time.perf_counter() - _t0) / X.shape[0])
                        if isinstance(preds, tuple):               # joint estimator: (doas, aux)
                            preds, aux = preds
                            preds, aux = preds.cpu(), aux.cpu()
                            span = sim.theta_range[1] - sim.theta_range[0]
                            if getattr(est, 'joint_axis', 'range') == 'range':
                                # ONE joint-normalized assignment; match in u = 1/r (bounded)
                                from cnmmusic.criteria.metrics import matched_tuple_rmse
                                u_max = 1.0 / min(sim.range_range) if sim.range_range else 0.2
                                r_true = b['ranges'][:, 0]
                                u_pred = torch.where(aux.double() > 0, 1.0 / aux.double(),
                                                     torch.zeros_like(aux.double()))
                                u_true = torch.where(torch.isfinite(r_true), 1.0 / r_true,
                                                     torch.zeros_like(r_true))
                                th_err, (r_err,) = matched_tuple_rmse(
                                    preds, doas, theta_span=span,
                                    aux=[(u_pred, u_true, u_max, aux.double(), r_true)])
                                errs.setdefault(f'{est_name}_range', []).append(r_err)
                                errs[est_name].append(th_err)
                                continue
                            if aux.ndim == 2 and b['freqs'].ndim == 2:   # per-source (tones)
                                from cnmmusic.criteria.metrics import matched_tuple_rmse
                                f_span = (max(sim.freq_range) - min(sim.freq_range)) or 0.4
                                f_true = b['freqs'][:, :aux.shape[-1]].double()
                                th_err, (f_err,) = matched_tuple_rmse(
                                    preds, doas, theta_span=span,
                                    aux=[(aux.double(), f_true, f_span)])
                                errs.setdefault(f'{est_name}_freq', []).append(100.0 * f_err)
                                errs[est_name].append(th_err)
                                continue
                            if aux.ndim == 2:                      # per-source est, shared truth
                                aux = aux.mean(-1)
                            if b['freqs'].ndim == 2:
                                # the scene has a carrier PER SOURCE but this estimator reports
                                # a single shared one: the frequency read-out does not apply
                                errs.setdefault(f'{est_name}_freq', []).append(
                                    torch.full((preds.shape[0],), torch.nan,
                                               dtype=torch.float64))
                            else:                       # shared carrier: scalar err in % f_c
                                errs.setdefault(f'{est_name}_freq', []).append(
                                    100.0 * (aux.double() - b['freqs']).abs())
                        preds = preds.cpu()
                        if torch.isfinite(preds).all():
                            errs[est_name].append(rmspe(preds, doas, reduce=False))
                        else:                     # estimator not applicable at this operating
                            errs[est_name].append(  # point (it reports non-finite DoAs)
                                torch.full((preds.shape[0],), torch.nan, dtype=torch.float64))
                        continue
                    # unknown d: native estimators report their own count, others follow AIC
                    if getattr(est, 'native_d', False):
                        pred_pad, d_hat = est(X, n_src=None, **kwargs)
                    else:
                        d_hat = d_aic
                        M_arr = X.shape[-2]
                        pred_pad = torch.full((X.shape[0], M_arr - 1), torch.nan,
                                              dtype=torch.float64)
                        for dv in d_hat.unique():
                            idx = (d_hat == dv).nonzero(as_tuple=True)[0]
                            kw = {k: v[idx] for k, v in kwargs.items()}
                            pred_pad[idx, :int(dv)] = est(X[idx], n_src=int(dv), **kw)
                    from cnmmusic.criteria.metrics import rmspe_est_d
                    e, acc = rmspe_est_d(pred_pad.cpu(), d_hat.cpu(), doas)
                    errs[est_name].append(e)
                    errs.setdefault(f'{est_name}_dacc', []).append(acc)
            for est_name in list(errs):
                vals = errs[est_name]
                if vals[0].ndim > 0:               # per-scene errors -> RMS mean over all scenes
                    per_scene = torch.cat(vals)
                    # THE metric: per-sample RMSPE, averaged over scenes. The tail statistics
                    # of the same per-scene errors ride along as companions.
                    h5[f'{param}={val}/{est_name}'] = per_scene.mean().item()
                    h5[f'{param}={val}/{est_name}_median'] = per_scene.median().item()
                    h5[f'{param}={val}/{est_name}_p99'] = per_scene.quantile(0.99).item()
                    h5[f'{param}={val}/{est_name}_rms'] = per_scene.pow(2).mean().sqrt().item()
                else:                              # batch scalars (est-d path, count acc)
                    for suf in ('', '_median', '_p99', '_rms'):
                        h5[f'{param}={val}/{est_name}{suf}'] = torch.stack(vals).mean().item()
            for est_name, ts in secs.items():          # inference cost [ms per scene]
                if not ts:
                    continue
                t = 1e3 * torch.tensor(ts, dtype=torch.float64)
                h5[f'{param}={val}/{est_name}_ms'] = t.mean().item()
                h5[f'{param}={val}/{est_name}_ms_std'] = t.std().item() if len(t) > 1 else 0.0
                h5[f'{param}={val}/{est_name}_ms_se'] = (t.std() / len(t) ** 0.5).item() \
                    if len(t) > 1 else 0.0
                h5[f'{param}={val}/{est_name}_ms_min'] = t.min().item()
            h5.flush()

        return {f'{param}={v}': {e: h5[f'{param}={v}'][e][()] for e in h5[f'{param}={v}']}
                for v in values}


#**************#
#   plotting   #
#**************#
def plot_experiment(results, param, values, out_path=None, ax=None, ylabel='RMSPE [deg]',
                    exclude=('oracle',), to_deg=True):
    import numpy as np
    import matplotlib
    if out_path is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from cnmmusic.visualize import use_paper_style, save_fig, method_style, COL_W
    use_paper_style()
    methods = [m for m in {m for d in results.values() for m in d}
               if m not in (exclude or ())
               and not m.endswith(('_dacc', '_range', '_freq', '_median', '_p99', '_rms',
                                   '_ms', '_ms_std', '_ms_se', '_ms_min'))]
    methods = sorted(methods, key=lambda m: (m not in ('CRB', 'ZZB'), m))  # bounds adjacent
    if ax is None:
        fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    else:
        fig = ax.figure
    for i, m in enumerate(methods):
        y = [results[f'{param}={v}'].get(m, np.nan) for v in values]
        if to_deg:
            y = [np.rad2deg(v) for v in y]
        st = method_style(m, i)
        # z-order follows the legend order: the first-listed methods (bounds, then the CNM
        # family) draw ON TOP of the classical clutter instead of under it
        ax.semilogy(values, y, ms=2.5, lw=0.9, label=m, zorder=len(methods) - i + 2, **st)
    from cnmmusic.visualize import AXIS_LABELS
    ax.set_xlabel(AXIS_LABELS.get(param, param))
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.15, which='both')
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02, 1.0, 0.2), mode='expand',
              ncol=min(len(methods), 4), fontsize=5.5, handlelength=1.4,
              columnspacing=0.8, borderaxespad=0.0, frameon=False)
    if out_path is not None:
        import pathlib
        save_fig(fig, pathlib.Path(out_path).with_suffix(''))
        plt.close(fig)
    return ax
