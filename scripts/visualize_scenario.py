####################################################################################################
#                                       visualize_scenario.py                                      #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 22/07/26                                                                                #
#                                                                                                  #
# Purpose: Qualitative visualization of a trained CNM checkpoint on any scenario of the ladder     #
#          (cnmmusic/scenarios.py). Core figures for ALL scenarios: null-spectrum overlay (CNM     #
#          vs nominal MUSIC vs oracle, sphere-sampling spread band), the manifold correction       #
#          (needed vs applied vs left over + misalignment), and the PCA manifold portrait.         #
#          Scenario extras: severity axis (mismatch: rho | correlated: corr | freq: carrier |      #
#          nearfield: range band), joint (theta, r) / (theta, f) 2-D maps where the axis is        #
#          active, eigenvalue stems for correlated scenes, and the array geometry plot for         #
#          non-ULA scenarios. CPU only -- never touches a training GPU.                            #
#                                                                                                  #
#          Run: python scripts/visualize_scenario.py --ckpt <path> --scenario mismatch             #
#               [--out results/...] [--scenes 2] [--d 3] [--kappa 100]                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
import datetime
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from cnmmusic.config import DEFAULT_CONFIG, sim_config
from cnmmusic.scenarios import SCENARIOS
from cnmmusic.models.frameworkCNM import CNMFramework
from cnmmusic.data.simulator import NarrowbandSimulator
from cnmmusic.arrays.steering import nearfield_steering_matrix, gauge_fix
from cnmmusic.criteria.losses import subspace_ratio
from cnmmusic.visualize import (use_paper_style, plot_spectra_overlay,
                                plot_nearfield_spectrum, plot_freq_spectrum,
                                plot_eigenvalues, plot_array)

# unified colors across all figures: blue = nominal, green = true/oracle, black = CNM
C_NOM, C_TRUE, C_CNM, C_LEFT = '#0173B2', '#029E73', 'k', '#AA3377'

# per-scenario-family difficulty axis: (name, [(label, value), ...]); value semantics per family
AXES = {
    'mismatch': ('rho', [('rho0', 0.0), ('rho0.75', 0.75), ('rho1.5', 1.5)]),
    'correlated': ('corr', [('corr0', 0.0), ('corr0.9', 0.9), ('corr1', 1.0)]),
    'nearfield': ('range', [('near', (5.0, 15.0)), ('mid', (15.0, 40.0)),
                            ('far', (40.0, 100.0))]),
    'freq': ('freq', [('f0.8', 0.8), ('f1', 1.0), ('f1.2', 1.2)]),
    'plain': (None, [('', None)]),
}
CONDITIONS = (('easy', 15.0, 200), ('hard', 0.0, 20))            # (tag, SNR [dB], T)


def family(scenario):
    if scenario == 'mismatch':
        return 'mismatch'
    if scenario == 'correlated':
        return 'correlated'
    if 'nearfield' in scenario:
        return 'nearfield'
    if scenario.startswith('freq'):
        return 'freq'
    return 'plain'


#**********************#
#   batch generation   #
#**********************#
def build_simulator(scenario, axis_value, snr, T, d, seed):
    """Controlled scenes of one scenario: fixed condition, one point on the difficulty axis."""
    merged = {**DEFAULT_CONFIG, **SCENARIOS[scenario]}
    cfg = sim_config(merged)
    cfg.update(n_src_range=(d, d), snr_range=(snr, snr), snapshots_range=(T, T),
               log_snapshots=False, min_sep=math.radians(8.0),
               power_imbalance_range=(0.0, 0.0), segments=1, seed=seed)
    fam = family(scenario)
    if fam == 'mismatch':
        cfg['rho_range'] = (axis_value, axis_value)
    elif fam == 'correlated':
        cfg['source_corr_range'] = (axis_value, axis_value)
    elif fam == 'nearfield':
        cfg['range_range'] = tuple(axis_value)
    elif fam == 'freq':
        cfg['freq_range'] = (axis_value, axis_value)
    return NarrowbandSimulator(cfg)


#**********************#
#   scene quantities   #
#**********************#
@torch.no_grad()
def scene_quantities(model, batch, b, kappa=100.0, n_samples=16):
    """Per-scene tensors: grid spectra + gauge-fixed manifolds + latent (+ sphere spread)."""
    d = int(torch.as_tensor(batch['n_src']).reshape(-1)[b])
    B = batch['X'].shape[0]
    sub = {k: (v[b:b + 1] if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == B
               and k != 'grid' else v) for k, v in batch.items()}
    z = model.encode(sub)
    R = sub['R_hat'][:, 0]
    En = model.noise_subspace(R, d)
    grid = batch['grid']
    per_src_f = torch.is_tensor(batch['freqs']) and batch['freqs'].ndim == 2
    f_scene = 1.0 if per_src_f else float(torch.as_tensor(sub['freqs']).reshape(-1)[0])
    th = grid[None]
    rr = torch.full_like(th, torch.inf)
    ff = torch.full_like(th, f_scene)                # a_true_grid lives at the scene carrier
    pos = sub['positions']

    a_cnm = model.steer(z, pos, th, rr, ff)
    a_nom = nearfield_steering_matrix(pos, th, rr, ff)
    a_true = sub['a_true_grid']

    P = {'CNM-MUSIC': 1.0 / (subspace_ratio(En, a_cnm)[0] + 1e-12),
         'MUSIC': 1.0 / (subspace_ratio(En, a_nom)[0] + 1e-12),
         'Oracle': 1.0 / (subspace_ratio(En, a_true)[0] + 1e-12)}
    # Capon on the same covariance (diagonal loading as the mvdr back-end), nominal and
    # corrected manifold; peak-normalized like the null spectra
    Rl = R[0] + 1e-3 * R[0].diagonal().real.mean() * torch.eye(R.shape[-1], dtype=R.dtype)
    Rinv = torch.linalg.inv(Rl)
    for name, a in (('MVDR', a_nom[0]), ('CNM-MVDR', a_cnm[0])):
        an = a / a.abs().pow(2).sum(0, keepdim=True).sqrt().clamp_min(1e-12)
        P[name] = 1.0 / torch.einsum('mg,mk,kg->g', an.conj(), Rinv.to(an.dtype), an).real.clamp_min(1e-12)

    P_samp = None
    if model.hparams.backend == 'sphere':            # tangent-Gaussian vMF around the mean z
        gen = torch.Generator().manual_seed(0)
        draws = []
        for _ in range(n_samples):
            v = torch.randn(z.shape, generator=gen)
            v = v - (v * z).sum(-1, keepdim=True) * z
            zk = z + v / (kappa ** 0.5)
            zk = zk / zk.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            draws.append(1.0 / (subspace_ratio(En, model.steer(zk, pos, th, rr, ff)) + 1e-12))
        P_samp = torch.cat(draws, dim=0)

    doas = sub['doas'][0, 0, :d]
    r_true = sub['ranges'][0, 0, :d]
    f_true = (sub['freqs'][0, :d] if per_src_f
              else torch.full_like(doas, f_scene))
    J_truth = subspace_ratio(En, model.steer(z, pos, doas[None], r_true[None],
                                             f_true[None]))[0]
    return dict(z=z, d=d, grid=grid, doas=doas, r_true=r_true, f_true=f_true, En=En, R=R,
                pos=pos, a_cnm=gauge_fix(a_cnm)[0], a_nom=gauge_fix(a_nom)[0],
                a_true=a_true[0], P=P, P_samp=P_samp, J_truth=J_truth,
                snr=float(sub['snr'][0]), T=int(torch.as_tensor(sub['snapshots']).reshape(-1)[0]))


def scene_title(q, axis_name, axis_label):
    ax = f'{axis_label}, ' if axis_label else ''
    return rf'{ax}SNR = {q["snr"]:.0f} dB, T = {q["T"]}, d = {q["d"]}'


#******************#
#   core figures   #
#******************#
def fig_spectrum(q, title, path):
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    labels = list(q['P'])
    plot_spectra_overlay(ax, q['grid'], [q['P'][k] for k in labels], labels,
                         true_doas=q['doas'])
    styles = {'CNM-MUSIC': dict(ls='-', lw=1.4, zorder=2, color=C_CNM),
              'MUSIC': dict(ls='--', lw=1.1, zorder=3, color=C_NOM),
              'Oracle': dict(ls=':', lw=1.1, zorder=4, color=C_TRUE)}
    for line in ax.get_lines():                      # distinct dashes keep coincident curves
        if line.get_label() == 'true':               # visible (e.g. rho = 0)
            line.set_label('True')
        st = styles.get(line.get_label())
        if st is not None:
            line.set(linestyle=st['ls'], linewidth=st['lw'], zorder=st['zorder'],
                     color=st['color'])
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02, 1.0, 0.2), mode='expand',
              ncol=4, fontsize=5.5, handlelength=1.4, columnspacing=0.8,
              borderaxespad=0.0, frameon=False)
    if q['P_samp'] is not None:
        Ps = q['P_samp'].numpy()
        Ps = Ps / Ps.max(axis=1, keepdims=True)
        ang = np.rad2deg(q['grid'].numpy())
        ax.fill_between(ang, Ps.min(axis=0), Ps.max(axis=0), color='k', alpha=0.15, zorder=1)
    ax.set_title(title, fontsize=8, pad=28)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_manifold(q, title, path):
    """How much correction was needed vs applied vs left over, and what it buys."""
    M = q['a_nom'].shape[0]
    need = (q['a_true'] - q['a_nom']).norm(dim=0).numpy() / math.sqrt(M)
    applied = (q['a_cnm'] - q['a_nom']).norm(dim=0).numpy() / math.sqrt(M)
    left = (q['a_cnm'] - q['a_true']).norm(dim=0).numpy() / math.sqrt(M)
    ang = np.rad2deg(q['grid'].numpy())

    def align_err(a, b):
        num = np.abs((a.conj() * b).sum(axis=0)) ** 2
        den = (np.abs(a) ** 2).sum(axis=0) * (np.abs(b) ** 2).sum(axis=0)
        return 1.0 - num / np.clip(den, 1e-30, None)

    fig, axes = plt.subplots(2, 1, figsize=(5.2, 4.2), sharex=True,
                             gridspec_kw={'hspace': 0.1})
    axes[0].plot(ang, need, color=C_TRUE, lw=1.2, label=r'Needed $\|a_{\rm true} - a_0\|$')
    axes[0].plot(ang, applied, color=C_CNM, lw=1.2, label=r'Applied $\|\hat{a} - a_0\|$')
    axes[0].plot(ang, left, color=C_LEFT, lw=1.2,
                 label=r'Left over $\|\hat{a} - a_{\rm true}\|$')
    axes[0].set_ylabel('Correction size (per sensor)')
    axes[0].legend(fontsize=7, loc='upper center', ncol=3)
    axes[0].set_title(title, fontsize=8)
    at = q['a_true'].numpy()
    axes[1].plot(ang, align_err(at, q['a_nom'].numpy()), color=C_NOM, lw=1.2,
                 label='Nominal manifold')
    axes[1].plot(ang, align_err(at, q['a_cnm'].numpy()), color=C_CNM, lw=1.2, ls='--',
                 label='Corrected manifold')
    axes[1].set_yscale('log')
    axes[1].set_xlabel(r'$\theta$ [deg]')
    axes[1].set_ylabel(r'Misalignment vs $a_{\rm true}$')
    axes[1].legend(fontsize=7)
    for axx in axes:
        for a in np.rad2deg(q['doas'].numpy()):
            axx.axvline(a, color='#CC0000', ls='--', lw=0.8, alpha=0.6)
        axx.grid(True, alpha=0.15)
        axx.set_xlim(ang[0], ang[-1])
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_sensors(q, title, path):
    """Per-sensor gain and phase of the steering vector at every true DoA: nominal vs true
    vs corrected (all gauge-fixed to sensor 0), one column per source."""
    idx = [int(torch.argmin((q['grid'] - th).abs())) for th in q['doas']]
    M = q['a_nom'].shape[0]
    fig, axes = plt.subplots(2, len(idx), figsize=(2.2 * len(idx) + 0.6, 3.6),
                             sharex=True, squeeze=False)
    for j, (g, th) in enumerate(zip(idx, q['doas'])):
        for a, lab, c, mk in ((q['a_nom'][:, g], 'Nominal', C_NOM, 'o'),
                              (q['a_true'][:, g], 'True', C_TRUE, 's'),
                              (q['a_cnm'][:, g], 'CNM', C_CNM, 'x')):
            a = a.numpy()
            axes[0, j].plot(range(M), 20 * np.log10(np.abs(a)), mk, color=c, ms=4,
                            mfc='none', label=lab)
            axes[1, j].plot(range(M), np.degrees(np.angle(a)), mk, color=c, ms=4, mfc='none')
        axes[0, j].set_title(rf'$\theta$ = {math.degrees(float(th)):.1f}$^\circ$', fontsize=8)
        axes[1, j].set_xlabel('Sensor')
        axes[1, j].set_xticks(range(M))
    axes[0, 0].set_ylabel('Gain [dB]')
    axes[1, 0].set_ylabel('Phase [deg]')
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_portrait(q, title, path):
    """The three manifolds as curves in the true manifold's top-2 PC plane."""
    def emb(a):
        return np.concatenate([a.numpy().real, a.numpy().imag], axis=0)

    Et = emb(q['a_true'])
    mu = Et.mean(axis=1, keepdims=True)
    U, s, _ = np.linalg.svd(Et - mu, full_matrices=False)
    pct = 100.0 * s ** 2 / (s ** 2).sum()
    lo, hi = np.rad2deg(q['grid'][0].item()), np.rad2deg(q['grid'][-1].item())
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for name, a, color in (('Nominal', q['a_nom'], C_NOM),
                           ('CNM', q['a_cnm'], C_CNM),
                           ('True', q['a_true'], C_TRUE)):
        p = U[:, :2].T @ (emb(a) - mu)
        ax.plot(p[0], p[1], color=color, lw=1.0, label=name,
                ls={'Nominal': '-', 'CNM': '--', 'True': ':'}[name],
                alpha=0.9 if name != 'True' else 0.8)
        ax.scatter(p[0, ::60], p[1, ::60], s=8, color=color, zorder=3)
    idx = [int(np.argmin(np.abs(q['grid'].numpy() - t))) for t in q['doas'].numpy()]
    p = U[:, :2].T @ (emb(q['a_true']) - mu)
    ax.scatter(p[0, idx], p[1, idx], s=60, marker='x', color='#CC0000', lw=1.5,
               label='True DoAs', zorder=4)
    ax.annotate(rf'$\theta = {lo:.0f}^\circ$', (p[0, 0], p[1, 0]),
                textcoords='offset points', xytext=(6, 8), fontsize=7, color=C_TRUE)
    ax.annotate(rf'$\theta = +{hi:.0f}^\circ$', (p[0, -1], p[1, -1]),
                textcoords='offset points', xytext=(6, -12), fontsize=7, color=C_TRUE)
    ax.set_xlabel(f'PC 1 ({pct[0]:.0f}$\\%$ of variance)')   # $\%$ survives usetex AND
    ax.set_ylabel(f'PC 2 ({pct[1]:.0f}$\\%$)')               # the mathtext fallback
    ax.legend(fontsize=7)
    ax.set_title(f'Manifold portrait, {title}', fontsize=9)
    ax.grid(True, alpha=0.15)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


#*********************#
#   scenario extras   #
#*********************#
@torch.no_grad()
def _scan(model, q, th, rr, ff, nominal=False, chunk=4096):
    """Chunked null spectrum 1 / J over arbitrary tuple grids (corrected or nominal)."""
    out = []
    for s in range(0, th.shape[-1], chunk):
        a = (nearfield_steering_matrix(q['pos'], th[..., s:s + chunk], rr[..., s:s + chunk],
                                       ff[..., s:s + chunk]) if nominal
             else model.steer(q['z'], q['pos'], th[..., s:s + chunk], rr[..., s:s + chunk],
                              ff[..., s:s + chunk]))
        out.append(1.0 / (subspace_ratio(q['En'], a) + 1e-12))
    return torch.cat(out, dim=-1)


def fig_map2d(model, q, scenario, title, path, n_theta=241, n_2nd=48):
    """Joint (theta, r) or (theta, f) maps, CNM vs nominal side by side."""
    fam = 'nearfield' if SCENARIOS[scenario].get('range') else 'freq'
    grid = torch.linspace(float(q['grid'][0]), float(q['grid'][-1]), n_theta,
                          dtype=torch.float64)
    if fam == 'nearfield':
        lo, hi = SCENARIOS[scenario]['range']
        second = torch.logspace(math.log10(lo), math.log10(hi), n_2nd, dtype=torch.float64)
        th = grid.repeat(n_2nd)[None]
        rr = second.repeat_interleave(n_theta)[None]
        ff = torch.ones_like(th)
        plot_fn, truth2 = plot_nearfield_spectrum, q['r_true']
    else:                                            # freq family
        flo, fhi = SCENARIOS[scenario]['freq']
        second = torch.linspace(flo, fhi, n_2nd, dtype=torch.float64)
        th = grid.repeat(n_2nd)[None]
        rr = torch.full_like(grid.repeat(n_2nd)[None], torch.inf)
        ff = second.repeat_interleave(n_theta)[None]
        plot_fn, truth2 = plot_freq_spectrum, float(q['f_true'][0])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True,
                             gridspec_kw={'wspace': 0.08})
    for ax, nominal, name in ((axes[0], False, 'CNM'), (axes[1], True, 'Nominal')):
        P = _scan(model, q, th, rr, ff, nominal=nominal)[0].reshape(n_2nd, n_theta)
        im = plot_fn(ax, grid, second, P, title=name, true_doas=q['doas'],
                     **({'true_ranges': truth2} if fam == 'nearfield'
                        else {'true_f': truth2}))
    axes[1].set_ylabel('')
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label='[dB]')
    fig.suptitle(title, fontsize=8, y=1.04)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_eigvals(q, title, path):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    plot_eigenvalues(ax, q['R'][0], title=title)
    ax.axvline(q['d'] + 0.5, color='#CC0000', ls='--', lw=0.8)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_geometry(q, title, path):
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    plot_array(ax, q['pos'][0], title=title, doas=q['doas'])
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


#**********#
#   main   #
#**********#
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--scenario', required=True, choices=sorted(SCENARIOS))
    parser.add_argument('--out', default=None)
    parser.add_argument('--tag', default='')
    parser.add_argument('--scenes', type=int, default=2)
    parser.add_argument('--d', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--kappa', type=float, default=100.0)
    parser.add_argument('--n_samples', type=int, default=16)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    use_paper_style()
    out = args.out or os.path.join(
        'results', '{}_viz_{}{}'.format(datetime.date.today().isoformat(), args.scenario,
                                        f'_{args.tag}' if args.tag else ''))
    os.makedirs(out, exist_ok=True)

    model = CNMFramework.load_from_checkpoint(args.ckpt, map_location='cpu')
    model.eval()
    print('loaded:', {k: model.hparams[k] for k in ('encoder', 'backend', 'z_dim',
                                                    'theta_range')})

    fam = family(args.scenario)
    axis_name, values = AXES[fam]
    for vi, (vtag, value) in enumerate(values):
        for cond, snr, T in CONDITIONS:
            sim = build_simulator(args.scenario, value, snr, T, args.d,
                                  args.seed + 10 * vi)
            batch = sim.sample(args.scenes)
            for b in range(args.scenes):
                q = scene_quantities(model, batch, b, kappa=args.kappa,
                                     n_samples=args.n_samples)
                axis_label = rf'$\{axis_name}$ = {value}' if axis_name == 'rho' else \
                    (f'{axis_name} = {vtag[len(axis_name):] if vtag.startswith(axis_name) else vtag}'
                     if axis_name else '')
                title = scene_title(q, axis_name, axis_label)
                tag = '_'.join(filter(None, (vtag, cond, f'scene{b}')))
                fig_spectrum(q, title, f'{out}/spectrum_{tag}.png')
                print(f'{tag}: J at truth = {[f"{v:.4f}" for v in q["J_truth"].tolist()]}')
                fig_sensors(q, title, f'{out}/sensors_{tag}.png')
                fig_portrait(q, title, f'{out}/portrait_{tag}.png')
                if b == 0:
                    if SCENARIOS[args.scenario].get('range') or \
                            SCENARIOS[args.scenario].get('per_source_freq') or fam == 'freq':
                        fig_map2d(model, q, args.scenario, title, f'{out}/map2d_{tag}.png')
                    if fam == 'correlated':
                        fig_eigvals(q, title, f'{out}/eigvals_{tag}.png')
                    if SCENARIOS[args.scenario]['geometries'] != ['ula']:
                        fig_geometry(q, title, f'{out}/geometry_{tag}.png')
    print('figures ->', out)


if __name__ == '__main__':
    main()
