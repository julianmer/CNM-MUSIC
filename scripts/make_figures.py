####################################################################################################
#                                          make_figures.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/08/26                                                                                #
#                                                                                                  #
# Purpose: One driver for the paper figures: for every paper scenario,                             #
#          resolve the pinned training run's best checkpoint, sweep the scenario's ONE open        #
#          axis (CNM vs nominal baselines vs CRB, cached to HDF5), and compose the main-paper      #
#          panels -- the per-scenario overview grid and the joint 2-D flagship results --          #
#          plus appendix spectrum/landscape portraits at one hard scene each.                      #
#          Checkpoint inference only: never touches a training process.                            #
#                                                                                                  #
#          Run: python scripts/make_figures.py [--scenarios snr,mismatch] [--n_mc 1000]            #
#               [--runs mismatch=wandbid] [--figures all,overview] [--seed_from results/figures_vN]#
#               [--recompute CNM-MUSIC,CNM-MVDR] [--appendix] [--quick]                            #
#          Results are cached per (scenario, sweep value, method): a rerun into the same --out,    #
#          or into a new one seeded with --seed_from, recomputes nothing unless --recompute names  #
#          the methods whose runs have moved on.                                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
import datetime
import json
import math
import os
import pathlib
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_scenario import scenario_sim_config, pinned

from cnmmusic.arrays.geometry import ArrayGeometry
from cnmmusic.config import DEFAULT_CONFIG
from cnmmusic.evaluate import runExperiment, plot_experiment
from cnmmusic.methods import build
from cnmmusic.scenarios import SCENARIOS
from cnmmusic.visualize import use_paper_style, save_fig, method_style, COL_W, DBL_W


#********************#
#   paper contents   #
#********************#
# scenario -> wandb run id of the pinned final-recipe training run; checkpoints live under
# logs/lightning_logs/<id>/checkpoints (None = not trained yet, panel is skipped);
# --runs scenario=id overrides without editing
RUNS = {
    'snr': 'qc0hkvlx',
    'separation': 'jgtoqmsr',
    'snapshots': 'ivq2rl5e',
    'sources': 't9kpou1d',
    'correlated': 'nhbxajm4',
    'colored_noise': 'e43nym5l',
    'mismatch': 'zabs9un2',
    'broadband_ofdm': 'qq9ijeqe',
    # aux-axis scenarios: the pin-aux model, theta-only, the fair line against the
    # DoA-only estimators. RUNS_JOINT holds their joint counterparts
    'broadband_tones': 'al4xyd4r',
    'broadband_ofdm_carrier': 'y0am6pdc',
    'nearfield': 'my54lget',
    'nearfield_mismatch': 'jijmvvxk',
    'distributed': '5ui6fys6',
    'nested_sources': 'j143n4dk',
    'sensor_failure': '4ip2wdgv',
}

# scenario -> {learned baseline: wandb run id}; a scenario whose trio is not banked yet
# simply draws no learned-baseline lines
BASELINE_RUNS = {
    'snr': {'damusic': '5ewiib3c', 'subspacenet': 'kdym2x9j', 'gridcnn': '8xim3z1l'},
    'snapshots': {'damusic': 'phjifjab', 'subspacenet': 'ajfjy1co', 'gridcnn': 'e0jwbpu9'},
    'sources': {'damusic': 'bcep2udg', 'subspacenet': 'da5a1fkm', 'gridcnn': 'wojenba8'},
    'separation': {'damusic': 't6vei6nq', 'subspacenet': 'aaw301xe', 'gridcnn': 'l2nm5j82'},
    'correlated': {'damusic': 'xe4pu4fo', 'subspacenet': 'mst0z1jt', 'gridcnn': 'jlb8gg1l'},
    'mismatch': {'damusic': 's2m2w9pd', 'subspacenet': 'j5huqrlf', 'gridcnn': 'yjppgfao'},
    'broadband_tones': {'damusic': 'nmit09er', 'subspacenet': '76b9x0uw',
                        'gridcnn': 'kyu2k87f'},
    'broadband_ofdm': {'damusic': 'yohm4sii', 'subspacenet': 'yobpbr3x',
                       'gridcnn': 'qjkwdqn9'},
    'nearfield_mismatch': {'damusic': 'qqys2jt0', 'subspacenet': 'jqo94zm6',
                           'gridcnn': 'k94ned13'},
    'broadband_ofdm_carrier': {'damusic': 'nnskvuih', 'subspacenet': '5xijcavd',
                               'gridcnn': '5g6h80k6'},
    'nearfield': {'damusic': 'ysxt39rz', 'subspacenet': 'y8bs1qcm', 'gridcnn': '7aysgge5'},
    'nested_sources': {'damusic': 'zdjvjftm', 'subspacenet': 'v7cb4aby',
                       'gridcnn': 'ommaqbql'},
    'distributed': {'damusic': '1sx1plfz', 'subspacenet': '8tdilh9z', 'gridcnn': 'spnudn8m'},
    'colored_noise': {'damusic': 'yf0i6u1a', 'subspacenet': 'fk3or6sq', 'gridcnn': 'nlc2xpbl'},
    'sensor_failure': {'damusic': 'ifx03vgk', 'subspacenet': 'f0s2o1es', 'gridcnn': 'ec2b5i9a'},
}

# scenario -> wandb run id of the JOINT run: the same recipe with the aux axis left active
# in loss and scan, so the model estimates range / carrier alongside theta. Swept separately
# (cached as '<scenario>_joint') and read only by the aux-estimation figure
RUNS_JOINT = {
    'broadband_tones': 'rbfry4k8',
    'broadband_ofdm_carrier': 'fchwpz64',
    'nearfield': 'tex2ubxi',
    'nearfield_mismatch': 'x54rllr1',
}

# the classical joint scans the joint runs are measured against, per aux axis
AUX_BASELINES = {
    'nearfield': ['2d-music', '2d-mvdr', 'cascade', 'cascade-mvdr'],
    'nearfield_mismatch': ['2d-music', '2d-mvdr', 'cascade', 'cascade-mvdr'],
    # on a ULA the spatial manifold depends only on the product f sin(theta), so the scanned
    # methods below sit on that ambiguity ridge; ESPRIT (JAFE) resolves it from the temporal
    # shift invariance and is the identifiable competitor
    'broadband_tones': ['f-music', 'f-mvdr', 'f-cascade', 'f-cascade-mvdr', 'f-esprit'],
    'broadband_ofdm_carrier': ['f-music', 'f-mvdr', 'f-cascade', 'f-cascade-mvdr',
                               'f-esprit'],
}

# scenario -> (param, values): the scenario's ONE open axis, everything else at the base
# point. Where the axis physically continues, values extend PAST the trained span (dashed
# boundary in the panels): the OOD probes for the uncertainty-gated CNM line
SWEEPS = {
    'snr': ('snr', [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]),
    'separation': ('delta', [2, 3.5, 5, 6.5, 8, 9.5, 11, 12.5, 14, 15.5, 17, 18.5, 20]),
    'snapshots': ('snapshots', [2, 10, 30, 50, 100, 150, 200, 300, 400, 800, 1600]),
    'sources': ('n_src', [1, 2, 3, 4, 5]),
    'correlated': ('source_corr', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    'colored_noise': ('noise_corr', [0.0, 0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72,
                                     0.81, 0.9, 0.99]),
    'mismatch': ('rho', [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5,
                         2.75, 3.0]),
    'broadband_tones': ('f_center', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    'broadband_ofdm': ('subcarriers', [10, 30, 60, 100, 200, 400, 700, 1000]),
    'nearfield_mismatch': ('rho', [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25,
                                   2.5, 2.75, 3.0]),
    'broadband_ofdm_carrier': ('f_center', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    'nearfield': ('range', [5, 8, 12, 17, 23, 30, 38, 49]),
    'nested_sources': ('n_src', [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
    'distributed': ('spread_deg', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    'sensor_failure': ('n_failed', [0, 1, 2, 3, 4]),
}

# swept param -> SCENARIOS key holding the trained span (params absent here never leave it)
PARAM_KEY = {'snr': 'snr', 'snapshots': 'snapshots', 'rho': 'rho', 'spread_deg': 'spread_deg',
             'n_failed': 'n_failed'}

# the main-paper grid: how much / how good the data is (top), how the sources sit (bottom);
# the remaining scenarios live in the extensions and all-scenario figures
# three nested overview grids: small (the headline six), medium (the ALL opening six plus
# the three flagship joint / mismatch axes), large (every ALL axis except the three whose
# story a companion panel already tells)
OVERVIEW_S = ['snr', 'snapshots', 'mismatch',
              'separation', 'correlated', 'sources']
OVERVIEW_M = ['snr', 'snapshots', 'colored_noise',
              'sources', 'separation', 'correlated',
              'mismatch', 'nearfield', 'broadband_ofdm_carrier']
OVERVIEW_L = ['snr', 'snapshots', 'colored_noise',
              'sources', 'separation', 'correlated',
              'mismatch', 'sensor_failure', 'nested_sources',
              'distributed', 'nearfield', 'broadband_ofdm_carrier']

# panel companions: the SAME (deliberately rich) classical set on every overview panel --
# lines are cheap and pruning is easier than regretting. The joint scenarios use the 2-D
# nominal scans plus their classical near-field/broadband stable.
BASELINES = {'default': ['music', 'root-music', 'mvdr', 'ss-music', 'mle', 'crb'],
             'broadband_tones': ['music', 'root-music', 'mvdr', 'ss-music'],
             'broadband_ofdm_carrier': ['music', 'root-music', 'mvdr', 'ss-music', 'mle'],
             'nearfield_mismatch': ['music', 'root-music', 'mvdr', 'ss-music'],
             'nearfield': ['music', 'root-music', 'mvdr', 'ss-music', 'mle'],
             'nested_sources': ['music', 'root-music', 'mvdr', 'ss-music'],
             'distributed': ['music', 'root-music', 'mvdr', 'ss-music'],
             'sensor_failure': ['music', 'root-music', 'mvdr', 'ss-music']}

# panels sharing a swept parameter carry the signal model in their axis label
XLABEL = {
    'nearfield_mismatch': r'Nearfield Mismatch Severity $\rho$',
    'broadband_tones': r'Tone Carrier Frequency $f / f_c$',
    'broadband_ofdm_carrier': r'OFDM Carrier Frequency $f / f_c$',
    'nested_sources': r'Coarray Number of Sources $D$',
}

LOG_X = ()                        # every axis is linear (DA-MUSIC plots T linearly)


#*****************#
#   checkpoints   #
#*****************#
SNAPSHOT_DIR = None     # set by main: frozen copies of the evaluated checkpoints


def best_checkpoint(run_id):
    """
    The best-doa checkpoint (callback monitors val/music/doa_rmspe_deg); last.ckpt fallback.
    A live run rotates its single best checkpoint under us, so the resolved file is frozen
    as a copy in SNAPSHOT_DIR (the figure then records exactly what it evaluated).
    """
    ckpts = pathlib.Path('logs/lightning_logs') / run_id / 'checkpoints'
    best = sorted((p for p in ckpts.glob('*.ckpt') if not p.name.startswith('last')),
                  key=lambda p: p.stat().st_mtime)   # 'last.ckpt' and the resume's 'last-v1'
    best = best[-1] if best else ckpts / 'last.ckpt'
    if SNAPSHOT_DIR is None:
        return best
    frozen = SNAPSHOT_DIR / f'{run_id}-{best.name}'
    if not frozen.exists():
        shutil.copy(best, frozen)
    return frozen


def prune_cache(out, scenario, methods):
    """Drop the cached entries of `methods` (display names, every suffix) for one scenario."""
    import h5py
    path = out / 'cache' / f'{scenario}.h5'
    if not path.exists():
        return
    with h5py.File(path, 'a') as h5:
        for group in list(h5.keys()):
            for key in list(h5[group].keys()):
                if any(key == m or key.startswith(m + '_') for m in methods):
                    del h5[group][key]


#***********#
#   sweep   #
#***********#
def sweep_scenario(scenario, ckpt, out, n_mc, mc_batch, seed, device='cpu'):
    """CNM + baselines over the scenario's open axis; cached per (value, method) in out/cache."""
    scen = SCENARIOS[scenario]
    param, values = SWEEPS[scenario]
    fov = math.radians(scen['fov_deg'])                # ULA paper scenarios: +-60 deg
    # coarray scenarios hand every estimator the virtual ULA the simulator emits
    geom = ArrayGeometry.ula(scen.get('M_virtual') or scen['M'])
    # estimators scan past the FOV (same margin as training) so edge nulls are whole;
    # truths stay inside the FOV (the simulator uses fov via scenario_sim_config)
    scan = fov + math.radians(DEFAULT_CONFIG['scan_margin_deg'])
    # every scanned method gets the SAME grid as training (config grid_size, widened by the
    # scan margin): a finer evaluation scan for the classical spectra alone would be unfair.
    # GridCNN keeps its own trained grid (its output layer IS the grid) and Root-MUSIC /
    # ESPRIT / SubspaceNet are gridless by construction.
    est_cfg = dict(grid_size=int(round(DEFAULT_CONFIG['grid_size'] * scan / fov)),
                   theta_range=(-scan, scan), device=device)
    # CNM-MVDR is the transfer demonstration: the corrected manifold under a CLASSICAL
    # back-end. It does NOT transfer to a LEARNED readout -- DA-MUSIC's MLP was trained on
    # nominal-manifold spectra, so substituting the manifold puts it off-distribution
    # (measured ~31 deg, i.e. chance). The 'cnm-damusic' spec stays in methods.py, unplotted.
    key = scenario
    specs = ([f'cnm:{ckpt}', f'cnm-mvdr:{ckpt}']
             + [f'{m}:{best_checkpoint(rid)}'
                for m, rid in BASELINE_RUNS.get(key, {}).items()]
             + BASELINES.get(key, BASELINES['default']))
    ests = {display: est for display, est, _ in (build(s, geom, est_cfg) for s in specs)}
    base = pinned(scenario_sim_config(scen, fov), scen)
    return runExperiment(scenario, param, values, ests, base_config=base, n_mc=n_mc,
                         mc_batch=mc_batch, cache_dir=str(out / 'cache'), seed=seed,
                         device=device)


#*****************#
#   joint sweep   #
#*****************#
def sweep_joint(scenario, ckpt, out, n_mc, mc_batch, seed, device='cpu'):
    """
    The joint run on the scenario's own axis: CNM with the aux axis active, against the
    classical joint scans. Cached as '<scenario>_joint' so it never mixes with the DoA
    sweep, and read only by the aux-estimation figure.
    """
    scen = SCENARIOS[scenario]
    param, values = SWEEPS[scenario]
    fov = math.radians(scen['fov_deg'])
    geom = ArrayGeometry.ula(scen.get('M_virtual') or scen['M'])
    scan = fov + math.radians(DEFAULT_CONFIG['scan_margin_deg'])
    est_cfg = dict(grid_size=int(round(DEFAULT_CONFIG['grid_size'] * scan / fov)),
                   theta_range=(-scan, scan), device=device)
    # the classical joint scans get EXACTLY the CNM scan: the same aux span the scenario
    # draws from, at the same resolution on every axis (grid_size points, logspace in range
    # and linear in carrier, as cnmmusic.models.frameworkCNM.estimate builds them)
    if scen.get('range'):
        lo, hi = scen['range']
        est_cfg['range_grid'] = (float(lo), float(hi), est_cfg['grid_size'])
    if tuple(scen.get('freq', (1.0, 1.0))) != (1.0, 1.0):
        lo, hi = scen['freq']
        est_cfg['f_grid'] = (float(lo), float(hi), est_cfg['grid_size'])
    specs = [f'cnm:{ckpt}', f'cnm-mvdr:{ckpt}'] + AUX_BASELINES.get(scenario, [])
    ests = {display: est for display, est, _ in (build(s, geom, est_cfg) for s in specs)}
    base = pinned(scenario_sim_config(scen, fov), scen)
    return runExperiment(f'{scenario}_joint', param, values, ests, base_config=base,
                         n_mc=n_mc, mc_batch=mc_batch, cache_dir=str(out / 'cache'),
                         seed=seed, device=device)


def aux_results(res, suffix):
    """Slice '{method}{suffix}' entries into a plain per-method results dict."""
    return {k: {m[:-len(suffix)]: v for m, v in d.items() if m.endswith(suffix)}
            for k, d in res.items()}


#************#
#   panels   #
#************#
def in_distribution(scenario, param, values):
    """The swept values inside the trained span (all of them where the axis never leaves it)."""
    span = SCENARIOS[scenario].get(PARAM_KEY[param]) if param in PARAM_KEY else None
    return [v for v in values if span is None or span[0] <= v <= span[1]]


def mark_trained_span(ax, scenario, param, values):
    """Dashed boundary where the eval sweep crosses an edge of the trained span."""
    span = SCENARIOS[scenario].get(PARAM_KEY[param]) if param in PARAM_KEY else None
    for edge in span or ():
        if min(values) < edge < max(values):
            ax.axvline(edge, color='0.65', lw=0.6, ls='--', zorder=1)


def shared_legend(fig, axes):
    """Replace per-axes legends with ONE single-row figure legend (methods match across
    panels; every canvas is overview-wide, so one row always fits like the overview's)."""
    handles, labels = [], []
    for ax in axes:
        if ax.get_legend():
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi), labels.append(li)
            ax.get_legend().remove()
    if handles:
        # 7 pt is the largest that still fits the full method row (10 entries) in one row
        # on the DBL_W canvas once the handles and column gaps are tightened to match
        fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=7,
                   handlelength=1.2, columnspacing=0.6, frameon=False)


# the overview's six panel boxes on the (DBL_W, 4.4) canvas (2x3, legend rect 0.95),
# frozen so EVERY figure places its panels at exactly these positions -- no per-figure
# re-layout, so all panels are pixel-identical to the overview's
# frozen panel geometry [inches]: column lefts and width, row pitch and height, and the
# legend headroom above the top row -- every figure gets identically sized panels
PANEL_X, PANEL_W = (0.0664 * DBL_W, 0.3869 * DBL_W, 0.7075 * DBL_W), 0.2757 * DBL_W
PANEL_H, ROW_PITCH, HEAD = 0.3649 * 4.4, (0.5578 - 0.0965) * 4.4, 4.4 - (0.5578 + 0.3649) * 4.4
BOTTOM = 0.0965 * 4.4


def panel_grid(n, per_row=3):
    """n panels on the frozen grid, reading order; a lone panel in a row sits in the middle
    slot, a pair sits half a cell right of the left slots (centered). Empty regions hold
    no artists, so the tight save crops them away."""
    rows = -(-n // per_row)
    height = HEAD + PANEL_H + (rows - 1) * ROW_PITCH + BOTTOM
    fig = plt.figure(figsize=(DBL_W, height))
    half = (PANEL_X[1] - PANEL_X[0]) / 2
    axes = []
    for i in range(n):
        r, k = divmod(i, per_row)
        rem = min(n - per_row * r, per_row)
        x = PANEL_X[k + 1 if (rem == 1 and k == 0) else k] + (half if rem == 2 else 0.0)
        y = BOTTOM + (rows - 1 - r) * ROW_PITCH
        axes.append(fig.add_axes((x / DBL_W, y / height, PANEL_W / DBL_W, PANEL_H / height)))
    return fig, axes


def finish(fig, path):
    """Single-row legend plus a white full-width line under it: the tight crop then keeps
    the full canvas width on every figure."""
    shared_legend(fig, fig.axes)
    fig.add_artist(matplotlib.lines.Line2D([-0.007, 1.014], [0.97, 0.97], color='white', lw=0.1,
                                           transform=fig.transFigure))
    # bottom pin below the lowest panel row: figures with equal row counts crop to
    # exactly the same height regardless of their label descenders
    low = min(ax.get_position().y0 for ax in fig.axes) - 0.1058
    fig.add_artist(matplotlib.lines.Line2D([0.45, 0.55], [low, low], color='white',
                                           lw=0.1, transform=fig.transFigure))
    save_fig(fig, path)
    plt.close(fig)


def compose_overview(results, out, scenarios=None, name='overview'):
    """Overview grid: one panel per scenario, RMSPE over the scenario's axis."""
    scenarios = OVERVIEW_S if scenarios is None else scenarios
    ylabel = 'RMSPE [deg]'
    fig, axes = panel_grid(len(scenarios))
    for i, (scenario, ax) in enumerate(zip(scenarios, axes)):
        if scenario not in results:
            ax.axis('off')
            ax.text(0.5, 0.5, f"{scenario.replace('_', ' ')}\n(no run yet)", ha='center',
                    va='center', fontsize=7, transform=ax.transAxes)
            continue
        param, values = SWEEPS[scenario]
        values = in_distribution(scenario, param, values)      # OOD tails: see compose_ood
        plot_experiment(results[scenario], param, values, ax=ax, ylabel=ylabel)
        ax.set_xlabel(XLABEL.get(scenario, ax.get_xlabel()))   # the scenario's own axis name
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if scenario == 'nested_sources':      # integer coarray count: major ticks only
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if param in LOG_X:
            ax.set_xscale('log')
        if param == 'n_src':
            ax.set_xticks(values)
        if i % 3:
            ax.set_ylabel('')
    finish(fig, out / name)


def compose_speed(results, out, device):
    """
    Inference cost, harvested from the sweeps themselves (no extra passes): per scene within
    each sweep point, pooled per scenario, and pooled overall. Writes speed.json and a bar
    chart ordered by the overall median-of-scenario means.
    """
    per_scen, methods = {}, set()
    for scenario, res in results.items():
        if scenario.endswith('_joint'):     # the DoA methods only; the joint runs are timed
            continue                        # on a different task and would double-count CNM
        acc = {}
        for point in res.values():
            for k, v in point.items():
                if k.endswith('_ms'):
                    acc.setdefault(k[:-3], []).append(float(v))
        if acc:
            per_scen[scenario] = {m: {'mean_ms': float(np.mean(v)),
                                      'std_ms': float(np.std(v)),
                                      'se_ms': float(np.std(v) / max(len(v), 1) ** 0.5),
                                      'n_points': len(v)} for m, v in acc.items()}
            methods |= set(acc)
    if not per_scen:
        return
    overall = {}
    for m in methods:
        vals = [d[m]['mean_ms'] for d in per_scen.values() if m in d]
        overall[m] = {'mean_ms': float(np.mean(vals)), 'std_ms': float(np.std(vals)),
                      'se_ms': float(np.std(vals) / max(len(vals), 1) ** 0.5),
                      'n_scenarios': len(vals)}
    json.dump({'device': device, 'per_scenario': per_scen, 'overall': overall},
              open(out / 'speed.json', 'w'), indent=2)

    order = sorted(overall, key=lambda m: overall[m]['mean_ms'])
    fig, ax = plt.subplots(figsize=(COL_W, 0.19 * len(order) + 0.9))
    y = np.arange(len(order))
    ax.barh(y, [overall[m]['mean_ms'] for m in order],
            xerr=[overall[m]['se_ms'] for m in order],
            color=[method_style(m).get('color', 'C0') for m in order], height=0.7)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=6)
    ax.invert_yaxis(); ax.set_xscale('log')
    ax.set_xlabel(f'Inference time [ms per scene, {device}]')
    ax.grid(True, axis='x', alpha=0.15, which='both')
    fig.tight_layout()
    save_fig(fig, out / 'speed')
    plt.close(fig)
    print(f'  speed [{device}] ms/scene, pooled over scenarios:')
    for m in order:
        o = overall[m]
        print(f"    {m:14s} {o['mean_ms']:8.3f} +- {o['se_ms']:.3f} (se)  "
              f"over {o['n_scenarios']} scenarios")
def compose_ood(results, out):
    """Same sweeps drawn over their full extent, for the axes that leave the trained span."""
    scenarios = [s for s in results if not s.endswith('_joint')
                 and len(in_distribution(s, *SWEEPS[s])) < len(SWEEPS[s][1])][:3]
    if not scenarios:
        return
    fig, axes = panel_grid(len(scenarios))
    for scenario, ax in zip(scenarios, axes):
        param, values = SWEEPS[scenario]
        plot_experiment(results[scenario], param, values, ax=ax)
        mark_trained_span(ax, scenario, param, values)
        if param in LOG_X:
            ax.set_xscale('log')
        if ax is not axes[0]:
            ax.set_ylabel('')
    finish(fig, out / 'ood_overview')


# every swept scenario, one theme per row: data, sources, the array, the propagation model,
# broadband signals
ALL = ['snr', 'snapshots', 'colored_noise',
       'sources', 'separation', 'correlated',
       'mismatch', 'sensor_failure', 'nested_sources',
       'distributed', 'nearfield', 'nearfield_mismatch',
       'broadband_tones', 'broadband_ofdm', 'broadband_ofdm_carrier']


def compose_all(results, out):
    """Every swept scenario on one grid, in ALL order."""
    scenarios = [s for s in ALL if s in results]
    if not scenarios:
        return
    fig, axes = panel_grid(len(scenarios))
    for i, (scenario, ax) in enumerate(zip(scenarios, axes)):
        param, values = SWEEPS[scenario]
        values = in_distribution(scenario, param, values)
        plot_experiment(results[scenario], param, values, ax=ax)
        ax.set_xlabel(XLABEL.get(scenario, ax.get_xlabel()))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if scenario == 'nested_sources':      # integer coarray count: major ticks only
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if param == 'n_src':
            ax.set_xticks(values[::2] if len(values) > 6 else values)
        if i % 3:
            ax.set_ylabel('')
    finish(fig, out / 'all')


def compose_aux(results, out):
    """
    The second-quantity read-outs, ONLY against methods that actually estimate them --
    aux_results keeps just the methods with a '{method}_freq' / '{method}_range' record.
    Frequency panels on top, range panels below (their x-axes identify them).
    """
    rng = [(f'{s}_joint', '_range') for s in ('nearfield', 'nearfield_mismatch')
           if f'{s}_joint' in results]
    freq = [(f'{s}_joint', '_freq') for s in ('broadband_tones', 'broadband_ofdm_carrier')
            if f'{s}_joint' in results]
    keep = lambda sc, suf: any(v == v for d in aux_results(results[sc], suf).values()
                               for m, v in d.items() if m != 'CRB')
    freq = [x for x in freq if keep(*x)]
    rng = [x for x in rng if keep(*x)]
    if not freq and not rng:
        return
    fig, axes = panel_grid(len(rng) + len(freq), per_row=max(len(rng), len(freq)))
    for i, ((scenario, suf), ax) in enumerate(zip(rng + freq, axes)):
        param, values = SWEEPS[scenario.removesuffix('_joint')]
        yl = (r'$f$ RMSE [$\%$ of $f_c$]' if suf == '_freq'
              else r'Range RMSE [$\lambda/2$]')
        plot_experiment(aux_results(results[scenario], suf), param, values, ax=ax,
                        exclude=('CRB',), ylabel=yl, to_deg=False)
        if suf == '_freq':
            ax.set_xlabel(XLABEL.get(scenario.removesuffix('_joint'), ax.get_xlabel()))
        if i not in (0, len(rng)):
            ax.set_ylabel('')
    finish(fig, out / 'aux_estimation')
#**********#
#   main   #
#**********#
# figure name -> composer, for --figures
FIGURES = {
    'overview_s': lambda r, o, d: compose_overview(r, o, OVERVIEW_S, 'overview_s'),
    'overview_m': lambda r, o, d: compose_overview(r, o, OVERVIEW_M, 'overview_m'),
    'overview_l': lambda r, o, d: compose_overview(r, o, OVERVIEW_L, 'overview_l'),
    'ood_overview': lambda r, o, d: compose_ood(r, o),
    'all': lambda r, o, d: compose_all(r, o),
    'aux_estimation': lambda r, o, d: compose_aux(r, o),
    'speed': compose_speed,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', default=None,
                        help='comma list (default: every scenario with a pinned run)')
    parser.add_argument('--runs', default=None, help='overrides, e.g. mismatch=ab12cd34,...')
    parser.add_argument('--n_mc', type=int, default=1000)
    parser.add_argument('--mc_batch', type=int, default=250)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--out', default='results/figures')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', default='cpu', help="'cpu' or 'cuda': the"
                        ' device the estimators run on, and that the logged'
                        ' per-scene inference times refer to')
    parser.add_argument('--figures', default='all_figures',
                        help=f"comma list of {sorted(FIGURES)} or 'all_figures'")
    parser.add_argument('--seed_from', default=None,
                        help='previous --out whose cache is copied into an empty new --out')
    parser.add_argument('--add_baselines', default=None,
                        help="comma list of method specs appended to every scenario's baselines")
    parser.add_argument('--exclude', default=None,
                        help="comma list of method display names left out of every figure")
    parser.add_argument('--recompute', default=None,
                        help="comma list of method display names (e.g. 'CNM-MUSIC,CNM-MVDR')"
                        ' whose cached entries are dropped before sweeping')
    args = parser.parse_args()
    if args.add_baselines:                    # extra comparison lines, every scenario
        for specs in BASELINES.values():
            specs += [m for m in args.add_baselines.split(',') if m not in specs]

    torch.set_num_threads(args.threads)
    use_paper_style()
    runs, joint_runs = dict(RUNS), dict(RUNS_JOINT)
    if args.runs:
        runs.update(dict(kv.split('=') for kv in args.runs.split(',')))
    scenarios = (args.scenarios.split(',') if args.scenarios
                 else [s for s in SWEEPS if runs.get(s)])
    missing = [s for s in scenarios if not runs.get(s)]
    if missing:
        parser.error(f'no pinned run for: {missing} (fill RUNS or pass --runs)')

    out = pathlib.Path(args.out)
    (out / 'cache').mkdir(parents=True, exist_ok=True)
    if args.seed_from and not any((out / 'cache').glob('*.h5')):
        for f in pathlib.Path(args.seed_from, 'cache').glob('*.h5'):
            shutil.copy(f, out / 'cache' / f.name)
        print(f'cache seeded from {args.seed_from}')
    if args.recompute:
        for s in scenarios:
            prune_cache(out, s, args.recompute.split(','))
    global SNAPSHOT_DIR
    SNAPSHOT_DIR = out / 'checkpoints'
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    ckpts = {s: str(best_checkpoint(runs[s])) for s in scenarios}
    json.dump({'runs': {s: runs[s] for s in scenarios}, 'checkpoints': ckpts,
               'sweeps': {s: SWEEPS[s] for s in scenarios}, 'n_mc': args.n_mc,
               'seed': args.seed, 'grid_size': DEFAULT_CONFIG['grid_size'],
               'datetime': datetime.datetime.now().isoformat(timespec='seconds')},
              open(out / 'config.json', 'w'), indent=2)

    results = {}
    for scenario in scenarios:
        print(f'== {scenario}: {ckpts[scenario]}')
        results[scenario] = sweep_scenario(scenario, ckpts[scenario], out, args.n_mc,
                                           args.mc_batch, args.seed, args.device)
        if scenario in joint_runs:              # the aux-estimation counterpart
            jck = str(best_checkpoint(joint_runs[scenario]))
            print(f'== {scenario} (joint): {jck}')
            results[f'{scenario}_joint'] = sweep_joint(scenario, jck, out, args.n_mc,
                                                       args.mc_batch, args.seed, args.device)
    # every figure composes from ALL scenarios cached under --out, so a pass over a subset
    # of scenarios never blanks the panels swept by an earlier pass into the same directory
    for scenario in SWEEPS:
        for key in (scenario, f'{scenario}_joint'):
            if key not in results and (out / 'cache' / f'{key}.h5').exists():
                param, values = SWEEPS[scenario]
                results[key] = runExperiment(key, param, values, {},
                                             cache_dir=str(out / 'cache'))

    figures = list(FIGURES) if args.figures == 'all_figures' else args.figures.split(',')
    unknown = [f for f in figures if f not in FIGURES]
    if unknown:
        parser.error(f'unknown figures: {unknown} (known: {sorted(FIGURES)})')
    if args.exclude:                          # drop the methods (all suffixes) from the plots
        drop = args.exclude.split(',')
        for res in results.values():
            for point in res.values():
                for k in [k for k in point if any(k == m or k.startswith(m + '_') for m in drop)]:
                    del point[k]
    for name in figures:
        FIGURES[name](results, out, args.device)
    print(f'figures -> {out}')


if __name__ == '__main__':
    main()
