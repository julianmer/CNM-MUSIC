####################################################################################################
#                                         test_scenario.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: THE evaluation entry point: any scenario x any methods (learned ones take their         #
#          checkpoint as 'name:path.ckpt'). Writes one datetime-stamped run directory with the     #
#          full configuration, pinned per-axis sweeps + scenario-average summary (quantitative)    #
#          and spectra / roots / eigenvalue / geometry showcases (qualitative).                    #
#                                                                                                  #
#          python scripts/test_scenario.py --scenario narrowband \                                 #
#              --methods mvdr,music,root-music,esprit,mle,crb[,cnm-root:checkpoints/x.ckpt]        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
import csv
import datetime
import json
import math
import pathlib
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import linear_sum_assignment

from cnmmusic.arrays.geometry import ArrayGeometry
from cnmmusic.arrays.imperfections import ImperfectionModel
from cnmmusic.config import DEFAULT_CONFIG
from cnmmusic.arrays.steering import steering_matrix, steering_derivative
from cnmmusic.archs.music import DifferentiableMUSIC, DifferentiableRootMUSIC
from cnmmusic.criteria.metrics import wrapped_diff
from cnmmusic.evaluate import runExperiment, plot_experiment
from cnmmusic.methods import build
from cnmmusic.scenarios import SCENARIOS
from cnmmusic.utils.crb import stochastic_crb
from cnmmusic.data.simulator import NarrowbandSimulator
from cnmmusic.visualize import (use_paper_style, save_fig, plot_spectra_overlay, plot_roots,
                                plot_nearfield_spectrum, plot_freq_spectrum, COL_W)

use_paper_style()

# number of Monte-Carlo cases drawn per sweep point and for the scenario-average summary
# (--n_mc overrides from the command line)
N_MC = 1000

# reference operating point: non-swept axes are pinned here (stated in every caption);
# None = derive from the scenario ranges (arithmetic mean for SNR, geometric mean for the
# log-scaled snapshot and range axes, rounded mean for the source count)
REFERENCE = {'snr': None, 'snapshots': None, 'n_src': None}

# scan-grid resolution and the number of example conditions per qualitative topic
GRID = DEFAULT_CONFIG['grid_size']
N_TOPIC = 7


#*************#
#   helpers   #
#*************#
def scenario_sim_config(scen, fov):
    return {
        'M': scen['M'], 'geometries': scen['geometries'], 'spacing': scen['spacing'],
        'freq_range': tuple(scen['freq']), 'snr_range': tuple(scen['snr']),
        'snapshots_range': tuple(scen['snapshots']), 'log_snapshots': True,
        'n_src_range': tuple(scen['n_src']), 'min_sep': math.radians(scen['min_sep_deg']),
        'power_imbalance_range': tuple(scen['power_imbalance_db']),
        'source_corr_range': tuple(scen['source_corr']),
        'noise_corr_range': tuple(scen['noise_corr']) if scen.get('noise_corr') else None,
        'range_range': tuple(scen['range']) if scen.get('range') else None,
        'signal': scen.get('signal', 'gauss'),
        'per_source_freq': scen.get('per_source_freq', False),
        'subcarriers': int(scen.get('subcarriers') or 0),
        'bandwidth': float(scen.get('bandwidth') or 0.0),
        'imperfections': (ImperfectionModel(randomized=True) if scen.get('rho') else None),
        'rho_range': tuple(scen['rho']) if scen.get('rho') else None,
        'coarray': scen.get('coarray', False),
        'spread_deg': tuple(scen['spread_deg']) if scen.get('spread_deg') else None,
        'n_failed': tuple(scen['n_failed']) if scen.get('n_failed') else None,
        'theta_range': (-fov, fov), 'grid_size': GRID,
    }


def reference_point(scen):
    """Resolve the reference operating point: explicit values, or scenario-range means."""
    ref = {}
    ref['snr'] = (REFERENCE['snr'] if REFERENCE['snr'] is not None
                  else round(sum(scen['snr']) / 2, 1))
    ref['snapshots'] = (REFERENCE['snapshots'] if REFERENCE['snapshots'] is not None
                        else int(round(math.sqrt(scen['snapshots'][0] * scen['snapshots'][1]))))
    ref['n_src'] = (REFERENCE['n_src'] if REFERENCE['n_src'] is not None
                    else int(round(sum(scen['n_src']) / 2)))
    if scen.get('range'):
        ref['range'] = round(math.sqrt(scen['range'][0] * scen['range'][1]), 1)
    return ref


def pinned(base, scen):
    """Reference operating point on top of the scenario config."""
    ref = reference_point(scen)
    cfg = dict(base)
    cfg['snr_range'] = (ref['snr'], ref['snr'])
    cfg['snapshots_range'] = (ref['snapshots'], ref['snapshots'])
    cfg['n_src_range'] = (ref['n_src'], ref['n_src'])
    if 'range' in ref:
        cfg['range_range'] = (ref['range'], ref['range'])
    return cfg


def sweep_values(scen, fov_deg):
    """Per-axis sweep grids derived from the scenario's ranges."""
    lo, hi = scen['snr']
    sweeps = {'snr': [round(v) for v in np.linspace(lo, hi, 7)]}
    lo, hi = scen['snapshots']
    sweeps['snapshots'] = sorted({int(round(v)) for v in np.geomspace(max(lo, 1), hi, 7)})
    lo, hi = scen['n_src']
    sweeps['n_src'] = sorted({int(round(v)) for v in np.linspace(lo, min(hi, scen['M'] - 2), 7)})
    # exact 2-source separation [deg]: one decade up from the scenario's minimum separation
    lo = max(scen['min_sep_deg'], 1.0)
    sweeps['delta'] = sorted({int(round(v)) for v in np.linspace(lo, min(10.0 * lo, fov_deg), 7)})
    if scen.get('rho'):
        lo, hi = scen['rho']
        sweeps['rho'] = [round(float(v), 2) for v in np.linspace(lo, hi, 7)]
    lo, hi = scen['source_corr']
    if hi > lo:
        sweeps['source_corr'] = [round(float(v), 2) for v in np.linspace(lo, hi, 7)]
    if scen.get('noise_corr'):
        lo, hi = scen['noise_corr']
        sweeps['noise_corr'] = [round(float(v), 2) for v in np.linspace(lo, hi, 7)]
    lo, hi = scen['freq']
    if hi > lo:                                    # carrier offset as % off the lambda/2 spacing
        sweeps['freq'] = [round(float(v - 1.0) * 100, 1) for v in np.linspace(lo, hi, 7)]
    if scen.get('range'):
        lo, hi = scen['range']
        sweeps['range'] = sorted({int(round(v)) for v in np.geomspace(lo, hi, 6)})
    return sweeps


def fixed_geometry(kind, M, seed):
    """
    One representative array per kind. Mixture scenarios draw a fresh array per batch, but the
    classical estimators bake the manifold in at build time -- the per-aperture evaluation
    therefore pins ONE array per kind and gives the identical instance to simulator + estimators.
    """
    if kind == 'ula':
        return ArrayGeometry.ula(M)
    if kind == 'uca':
        return ArrayGeometry.uca(M, radius=1.0 / (2 * math.sin(math.pi / M)))
    if kind == 'ura':
        mx = max(2, int(M ** 0.5))
        return ArrayGeometry.ura(mx, max(M // mx, 1))
    rng = torch.Generator().manual_seed(seed)
    if kind == 'nula':
        gaps = 0.5 + torch.rand(M - 1, generator=rng, dtype=torch.float64)
        p = torch.cat([torch.zeros(1, dtype=torch.float64), gaps.cumsum(0)])
        return ArrayGeometry.nonuniform_la(p * (M - 1) / p[-1])
    if kind == 'random_planar':
        return ArrayGeometry.random_planar(M, float(M - 1), rng=rng)
    raise ValueError(f'unknown geometry kind: {kind}')


def marginal_summary(sim, ests, oracle_like, n_cases, batch, known_n_src, theta_range=None):
    """
    Scenario-average per-method error stats (RMSPE / mean / std / median, all in degrees over
    per-scene RMSEs) over one fully-drawn pool (all axes varying). Pool batches mix T and d per
    element, so scoring runs per dense (T, d) group. ZZB needs theta_range; its std is across
    per-group bound values (each already a small scene average), not single scenes.
    """
    from cnmmusic.models.framework import condition_groups, sub_batch
    from cnmmusic.utils.crb import nearfield_crb, freq_crb
    sq = {n: [] for n in ests if ests[n] is not None}
    sq_aux = {}                                # per-scene squared aux errors (range / freq)
    crb, zzb, crb_r, crb_f = [], [], [], []
    for _ in range(max(n_cases // batch, 1)):
        full = sim.sample(batch)
        for idx, T, d in condition_groups(full):
            b = sub_batch(full, idx, T, d)
            X, doas = b['X'][:, 0], b['doas'][:, 0]
            A = steering_matrix(b['positions'], doas)
            dA = steering_derivative(b['positions'], doas)
            v = stochastic_crb(A, dA, b['R_s'][:, 0], b['sigma2'], T).clamp_min(0.0)
            if theta_range is not None:            # clip at the uniform-prior bound (W^2/12)
                W = theta_range[1] - theta_range[0]
                v = v.clamp(max=W ** 2 / 12)
            crb.append(v.mean(-1))
            r_true = b['ranges'][:, 0]
            if torch.isfinite(r_true).all():
                _, vr = nearfield_crb(b['positions'][0], doas, r_true, b['R_s'][:, 0],
                                      b['sigma2'], T)
                if sim.range_range is not None:
                    rw = sim.range_range[1] - sim.range_range[0]
                    vr = vr.clamp(max=rw ** 2 / 12)
                crb_r.append(vr.mean(-1))
            if bool((b['freqs'] != 1.0).any()):
                _, vf = freq_crb(b['positions'][0], doas, b['freqs'], b['R_s'][:, 0],
                                 b['sigma2'], T)
                vf = vf.clamp(max=0.5 ** 2 / 12)   # F-MUSIC scan width as the prior
                crb_f.append(100.0 ** 2 * vf.mean(-1))
            if 'ZZB' in ests and theta_range is not None:
                from cnmmusic.utils.zzb import zzb_from_batch
                zzb.append(zzb_from_batch(b, b['positions'][0], theta_range, n_scenes=8))
            for n, e in ests.items():
                if e is None:
                    continue
                kwargs = ({'A_true': b['a_true_grid']} if n == 'oracle' else
                          {'imperfect': b['imperfect']} if n == 'oracle-root' else {})
                pred = e(X, n_src=d, **kwargs)
                aux = None
                if isinstance(pred, tuple):
                    pred, aux = pred
                    axis = getattr(e, 'joint_axis', 'range')
                for i in range(X.shape[0]):
                    cost = wrapped_diff(pred[i][:, None], doas[i][None, :]).pow(2)
                    ri, ci = linear_sum_assignment(cost.numpy())
                    sq[n].append(cost[ri, ci].mean())
                    if aux is None:
                        continue
                    if axis == 'range':                # angle-matched range error [lambda/2]
                        sq_aux.setdefault(n, []).append(
                            (aux[i][ri] - r_true[i][ci]).pow(2).mean())
                    else:                              # shared carrier error [% f_c]
                        sq_aux.setdefault(n, []).append(
                            (100.0 * (aux[i].double() - b['freqs'][i])) ** 2)

    raw = {n: np.rad2deg(torch.stack(v).sqrt().numpy()) for n, v in sq.items()}
    raw['CRB'] = np.rad2deg(torch.cat(crb).sqrt().numpy())
    if zzb:
        raw['ZZB'] = np.rad2deg(torch.stack(zzb).numpy())
    out = {n: summary_stats(v) for n, v in raw.items()}

    def aux_stats(prefix, v):
        return {f'{prefix}_mean': float(v.mean()), f'{prefix}_std': float(v.std()),
                f'{prefix}_median': float(np.median(v))}

    for n, v in sq_aux.items():
        prefix = ('range_rmse' if getattr(ests[n], 'joint_axis', 'range') == 'range'
                  else 'freq_rmse')
        out[n].update(aux_stats(prefix, torch.stack(v).sqrt().numpy()))
    if crb_r:
        out['CRB'].update(aux_stats('range_rmse', torch.cat(crb_r).sqrt().numpy()))
    if crb_f:
        out['CRB'].update(aux_stats('freq_rmse', torch.cat(crb_f).sqrt().numpy()))
    return out, raw


def summary_stats(v):
    """Mean/std/median [deg] of per-scene RMSE values."""
    return {'rmspe_mean_deg': float(v.mean()), 'rmspe_std_deg': float(v.std()),
            'rmspe_median_deg': float(np.median(v))}


def write_summary(summary, quant, extra_cols=()):
    json.dump(summary, open(quant / 'summary.json', 'w'), indent=2)
    with open(quant / 'summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        aux = [c for c in ('range_rmse_mean', 'range_rmse_std', 'range_rmse_median',
                           'freq_rmse_mean', 'freq_rmse_std', 'freq_rmse_median')
               if any(c in s for s in summary.values())]
        cols = ['rmspe_mean_deg', 'rmspe_std_deg', 'rmspe_median_deg'] + aux + list(extra_cols)
        writer.writerow(['method'] + cols)
        for name, s in summary.items():
            writer.writerow([name] + [f'{v:.4f}' if isinstance(v := s.get(c, ''), float) else v
                                      for c in cols])


#*********#
#   run   #
#*********#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', required=True, choices=sorted(SCENARIOS))
    parser.add_argument('--methods', required=True,
                        help="comma list; learned methods as 'name:checkpoint.ckpt'")
    parser.add_argument('--n_mc', type=int, default=N_MC)
    parser.add_argument('--known_n_src', default=None,
                        type=lambda v: str(v).lower() in ('1', 'true', 'yes'))
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    scen = SCENARIOS[args.scenario]
    known = (scen.get('known_n_src', True) if args.known_n_src is None else args.known_n_src)
    specs = [s.strip() for s in args.methods.split(',')]
    kinds = scen['geometries']
    per_kind = len(kinds) > 1                    # per-aperture mode: one full pass per array kind

    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run = pathlib.Path('results') / f"{stamp}_{args.scenario}{('_' + args.tag) if args.tag else ''}"
    run.mkdir(parents=True)
    json.dump({'scenario': args.scenario, 'scenario_config': {k: str(v) for k, v in scen.items()},
               'methods': specs,
               'checkpoints': {s.split(':')[0]: s.split(':', 1)[1] for s in specs if ':' in s},
               'apertures': kinds if per_kind else None,
               'n_mc': args.n_mc, 'seed': args.seed, 'known_n_src': known,
               'reference_point': reference_point(scen), 'grid_size': GRID,
               'datetime': stamp},
              open(run / 'config.json', 'w'), indent=2)

    pooled = {}                                  # per-method per-scene errors across apertures
    for kind in kinds:
        fov = math.radians({'ula': 60.0, 'nula': 60.0}.get(kind, 180.0))
        geom = fixed_geometry(kind, scen['M'], args.seed)
        # estimators scan past the FOV (same margin as training) so edge nulls are whole;
        # truths stay inside the FOV (the simulator uses fov below)
        scan = fov + math.radians(DEFAULT_CONFIG['scan_margin_deg'])
        est_cfg = dict(grid_size=int(round(GRID * scan / fov)), theta_range=(-scan, scan))
        built = [build(s, geom, est_cfg) for s in specs]
        ests = {display: est for display, est, _ in built}

        quant = run / 'quantitative' / kind if per_kind else run / 'quantitative'
        qual = run / 'qualitative' / kind if per_kind else run / 'qualitative'
        quant.mkdir(parents=True), qual.mkdir(parents=True)

        base = scenario_sim_config(scen, fov)
        if per_kind:                             # identical fixed array for data and estimators
            base['geometries'], base['geometry'] = None, geom
        sweeps = sweep_values(scen, math.degrees(fov))

        #*********************************************************************#
        #   quantitative: pinned per-axis sweeps + scenario-average summary   #
        #*********************************************************************#
        for param, values in sweeps.items():
            pin = pinned(base, scen)
            res = runExperiment(param, param, values, ests, n_mc=args.n_mc,
                                base_config=pin, cache_dir=str(quant), seed=args.seed,
                                known_n_src=known)
            ax = plot_experiment(res, param, values, out_path=None, exclude=())
            if param in ('snapshots', 'range'):
                ax.set_xscale('log')
            if param in ('n_src', 'delta'):
                ax.set_xticks(values)
            (quant / 'est_d').mkdir(exist_ok=True)
            save_fig(ax.figure, quant / 'est_d' / param)
            plt.close(ax.figure)

            # per estimated quantity: est_r (range [lambda/2]) and est_f (carrier [% f_c])
            for suffix, sub, ylab in (('_range', 'est_r', r'Range RMSE [$\lambda/2$]'),
                                      ('_freq', 'est_f', r'$f$ RMSE [$\%$ of $f_c$]')):
                res_aux = {k: {m[:-len(suffix)]: val for m, val in d.items()
                               if m.endswith(suffix)} for k, d in res.items()}
                if not any(res_aux.values()):
                    continue
                ax = plot_experiment(res_aux, param, values, out_path=None, exclude=(),
                                     ylabel=ylab, to_deg=False)
                if param in ('snapshots', 'range'):
                    ax.set_xscale('log')
                if param in ('n_src', 'delta'):
                    ax.set_xticks(values)
                (quant / sub).mkdir(exist_ok=True)
                save_fig(ax.figure, quant / sub / param)
                plt.close(ax.figure)

        sim = NarrowbandSimulator({**base, 'seed': args.seed + 999})
        summary, raw = marginal_summary(sim, ests, None, args.n_mc, 25, known,
                                        theta_range=(-fov, fov))
        write_summary(summary, quant)
        for name, v in raw.items():
            pooled.setdefault(name, []).append(v)

        #***************************************************************************************#
        #   qualitative: spectra/ and roots/ folders, one subfolder per topic, one figure per   #
        #***************************************************************************************#
        d_hi = min(scen['n_src'][1], scen['M'] - 2)
        snr_lo, snr_hi = scen['snr']
        T_lo, T_hi = scen['snapshots']
        pair = {'n_src_range': (2, 2), 'min_sep': math.radians(1.5), 'max_sep': math.radians(2.5)}
        topics = {
            'snr': [(f'snr_{v:+d}dB', {'snr_range': (v, v)})
                    for v in sorted({int(round(x)) for x in np.linspace(snr_lo, snr_hi, N_TOPIC)})],
            'snapshots': [(f'T_{v}', {'snapshots_range': (v, v)})
                          for v in sorted({int(round(x))
                                           for x in np.geomspace(max(T_lo, 1), T_hi, N_TOPIC)})],
            'sources': [(f'D_{d}', {'n_src_range': (d, d)}) for d in range(1, d_hi + 1)],
            'separation': [
                ('close_pair_reference', pair),
                ('close_pair_low_snr', {**pair, 'snr_range': (snr_lo + 5, snr_lo + 5)}),
                ('close_pair_high_snr', {**pair, 'snr_range': (snr_hi - 5, snr_hi - 5)}),
                ('close_pair_high_T', {**pair, 'snapshots_range': (T_hi, T_hi)}),
                ('close_pair_high_snr_T', {**pair, 'snr_range': (snr_hi - 5, snr_hi - 5),
                                           'snapshots_range': (T_hi, T_hi)}),
            ],
        }
        spectral = [(disp, est) for disp, est, _ in built if hasattr(est, 'spectrum')]
        rooter = DifferentiableRootMUSIC()

        # joint scenarios showcase the 2-D pseudo-spectrum maps instead of 1-D overlays
        f_lo, f_hi = scen['freq']
        has_r, has_f = bool(scen.get('range')), f_hi > f_lo
        if has_r:
            from cnmmusic.estimators.nearfield import NearFieldMUSIC
            nf_map = NearFieldMUSIC(geom, est_cfg)
        if has_f:
            from cnmmusic.estimators.classical import FreqMUSIC
            f_map = FreqMUSIC(geom, est_cfg)

        ex_i = 0                                        # fresh scene (angles) per example
        for topic, examples in topics.items():
            spec_dir, root_dir = qual / 'spectra' / topic, qual / 'roots' / topic
            if not (has_r or has_f):
                spec_dir.mkdir(parents=True, exist_ok=True)
            if kind == 'ula':
                root_dir.mkdir(parents=True, exist_ok=True)
            for label, override in examples:
                ex_i += 1
                sim = NarrowbandSimulator({**pinned(base, scen), **override,
                                           'seed': args.seed + 7 + 131 * ex_i})
                b = sim.sample(1)
                d = int(b['n_src'][0])
                R = b['R_hat'][:, 0]

                note = (f"$D={d}$" if label.startswith('D_')
                        else f"{label.replace('_', ' ')} ($D={d}$)")
                if has_r or has_f:                 # 2-D maps with truth marks (x)
                    map_dir = qual / 'maps' / topic
                    map_dir.mkdir(parents=True, exist_ok=True)
                    if has_r:
                        fig, ax = plt.subplots(figsize=(COL_W, 2.4))
                        plot_nearfield_spectrum(ax, nf_map.grid, nf_map.r_grid,
                                                nf_map.spectrum2d(R, d)[0],
                                                true_doas=b['doas'][0, 0],
                                                true_ranges=b['ranges'][0, 0])
                        ax.text(0.02, 0.96, note, transform=ax.transAxes, va='top',
                                fontsize=6, color='w')
                        save_fig(fig, map_dir / f'{label}_theta_r'); plt.close(fig)
                    if has_f:
                        fig, ax = plt.subplots(figsize=(COL_W, 2.4))
                        plot_freq_spectrum(ax, f_map.grid, f_map.f_grid,
                                           f_map.spectrum_map(b['X'][0, 0], d),
                                           true_doas=b['doas'][0, 0], true_f=b['freqs'][0])
                        ax.text(0.02, 0.96, note, transform=ax.transAxes, va='top',
                                fontsize=6, color='w')
                        save_fig(fig, map_dir / f'{label}_theta_f'); plt.close(fig)
                else:
                    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
                    spectra, labels, grid_ref = [], [], None
                    for disp, est in spectral:
                        P = (est.spectrum(b['X'][0, 0], d)[None]
                             if getattr(est, 'spectrum_on_snapshots', False)
                             else est.spectrum(R, d))
                        if P[0].ndim > 1:
                            continue
                        spectra.append(P[0])
                        labels.append(disp)
                        grid_ref = est.grid if grid_ref is None else grid_ref
                    if spectra:
                        plot_spectra_overlay(ax, grid_ref, spectra, labels,
                                             true_doas=b['doas'][0, 0])
                        ax.text(0.02, 0.96, note, transform=ax.transAxes, va='top',
                                fontsize=6, color='0.3')
                        save_fig(fig, spec_dir / label)
                    plt.close(fig)

                if kind == 'ula':
                    En = DifferentiableMUSIC(grid_size=GRID,
                                             theta_range=(-fov, fov)).noise_subspace(R, d)
                    roots = rooter.companion_roots(rooter.null_polynomial(En))[0]
                    mag = roots.abs()
                    dist = (1 - mag).abs() + torch.where(mag < 1, torch.zeros_like(mag),
                                                         torch.full_like(mag, 1e3))
                    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.9))
                    plot_roots(ax, roots, dist.topk(d, largest=False).indices,
                               true_doas=b['doas'][0, 0])
                    ax.text(0.02, 0.97, note, transform=ax.transAxes, va='top', fontsize=6,
                            color='0.3')
                    save_fig(fig, root_dir / label); plt.close(fig)

    if per_kind:                                 # general table: all apertures pooled per method
        overall = {}
        for name, chunks in pooled.items():
            v = np.concatenate(chunks)
            overall[name] = {**summary_stats(v), 'n_apertures': len(chunks)}
        write_summary(overall, run / 'quantitative', extra_cols=('n_apertures',))

    print(f'run written to {run}')
