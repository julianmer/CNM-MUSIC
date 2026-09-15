####################################################################################################
#                                       qualitative_figure.py                                      #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 28/08/26                                                                                #
#                                                                                                  #
# Purpose: Qualitative figures of a trained CNM checkpoint, rendered for many seeds and conditions #
#          so the scene can be chosen; checkpoint inference only. On the mismatch scenario, one    #
#          row of three panels on one drawn scene: the pseudo-spectra (nominal and corrected       #
#          manifold under MUSIC and Capon), the steering vector at one true DoA element by element #
#          as phasors, and a linear read-out of the mismatch severity from the encoder latent. On  #
#          the near-field and broadband-carrier scenarios, the joint (theta, r) or (theta, f) maps #
#          on the nominal and corrected manifold next to the estimated-versus-true read-out of the #
#          second parameter, as one figure or as its parts (--parts); --panels composes a custom   #
#          single-row layout.                                                                      #
#                                                                                                  #
#          Run: python scripts/qualitative_figure.py --ckpt <cnm.ckpt> [--scenario mismatch]       #
#               [--parts both|maps|readout] [--out results/qualitative]                            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_scenario import build_simulator, scene_quantities, _scan, C_NOM, C_TRUE, C_CNM
from cnmmusic.arrays.steering import nearfield_steering_matrix
from cnmmusic.criteria.losses import subspace_ratio

from cnmmusic.arrays.geometry import ArrayGeometry
from cnmmusic.config import DEFAULT_CONFIG, sim_config
from cnmmusic.data.simulator import NarrowbandSimulator
from cnmmusic.estimators.neural import SubspaceNetEstimator
from cnmmusic.methods import build
from cnmmusic.models.frameworkCNM import CNMFramework
from cnmmusic.scenarios import SCENARIOS
from cnmmusic.visualize import use_paper_style, method_style, DBL_W

# (tag, SNR [dB], T) of the drawn scene and the severities rendered
CONDITIONS = (('easy', 15.0, 200), ('hard', 0.0, 20))
RHOS = (1.0, 1.5)
# the latent panel: scenes at the base point over the trained severity span
LATENT_RHOS = np.linspace(0.0, 1.5, 7)
LATENT_SCENES = 500


#*************#
#   spectra   #
#*************#
def panel_spectra(ax, q, ssn=None, styled=False):
    """Peak-normalized pseudo-spectra in the method colors, solid or in the result plots'
    line styles: the classical back-ends on the nominal and corrected manifold, plus
    SubspaceNet's surrogate-covariance MUSIC; truths red dashed."""
    grid = np.degrees(q['grid'].numpy())
    ls = lambda name: method_style(name)['ls'] if styled else '-'
    # drawing order bottom -> top: MVDR, MUSIC, CNM-MVDR, CNM-MUSIC, then the truths
    for z, name in enumerate(('MVDR', 'MUSIC', 'CNM-MVDR', 'CNM-MUSIC'), start=1):
        P = q['P'][name].numpy()
        ax.plot(grid, P / P.max(), ls=ls(name), color=method_style(name)['color'], lw=1.1,
                label=name, zorder=z)
    if ssn is not None:
        g, P = ssn
        ax.plot(g, P / P.max(), ls=ls('SubspaceNet-MUSIC'),
                color=method_style('SubspaceNet-MUSIC')['color'], lw=1.1,
                label='SubspaceNet-MUSIC')
    for th in np.degrees(q['doas'].numpy()):
        ax.axvline(th, color='#CC0000', ls='--', lw=0.8, zorder=6)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_ylabel('Normalized Spectrum')


@torch.no_grad()
def ssn_spectrum(est, X, d):
    """SubspaceNet's MUSIC spectrum on its own scan grid -> (grid [deg], P)."""
    P = est.spectrum(X[None], n_src=d)[0]
    return np.degrees(est.grid.numpy()), P.numpy()


#**************#
#   manifold   #
#**************#
def steering_at(q, k):
    """Nominal, true and corrected steering vectors at the k-th true DoA (grid column)."""
    g = int(torch.argmin((q['grid'] - q['doas'][k]).abs()))
    return (q['a_nom'][:, g].numpy(), q['a_true'][:, g].numpy(), q['a_cnm'][:, g].numpy(),
            math.degrees(float(q['doas'][k])))


def panel_phasors(ax, q, k=0):
    """Each sensor's steering entry in the complex plane: nominal -> corrected, true marked."""
    a0, at, ac, th = steering_at(q, k)
    circle = np.exp(1j * np.linspace(0, 2 * np.pi, 200))
    ax.plot(circle.real, circle.imag, color='0.85', lw=0.6, zorder=0)
    for m in range(len(a0)):
        ax.annotate('', xy=(ac[m].real, ac[m].imag), xytext=(a0[m].real, a0[m].imag),
                    arrowprops=dict(arrowstyle='->', color='0.6', lw=0.6, shrinkA=2, shrinkB=2))
    ax.plot(a0.real, a0.imag, 'o', color=C_NOM, ms=3.5, mfc='none', label='Nominal')
    ax.plot(at.real, at.imag, 's', color=C_TRUE, ms=3.5, mfc='none', label='True')
    ax.plot(ac.real, ac.imag, 'x', color=C_CNM, ms=3.5, label='CNM')
    lim = 1.15 * max(np.abs(np.concatenate([a0, at, ac])).max(), 1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='datalim')      # equal scale in the square box
    ax.set_xlabel(rf'Re $a_m(\theta = {th:.1f}^\circ)$')
    ax.set_ylabel(r'Im $a_m$')


def panel_portrait(ax, q):
    """The three manifolds as curves over theta in the true manifold's top-2 PC plane, the
    true DoAs marked (the portrait of visualize_scenario.fig_portrait)."""
    emb = lambda a: np.concatenate([a.numpy().real, a.numpy().imag], axis=0)
    Et = emb(q['a_true'])
    mu = Et.mean(axis=1, keepdims=True)
    U, sv, _ = np.linalg.svd(Et - mu, full_matrices=False)
    pct = 100.0 * sv ** 2 / (sv ** 2).sum()
    for name, a, color, ls in (('Nominal', q['a_nom'], C_NOM, '-'), ('CNM', q['a_cnm'], C_CNM, '--'),
                               ('True', q['a_true'], C_TRUE, ':')):
        pr = U[:, :2].T @ (emb(a) - mu)
        ax.plot(pr[0], pr[1], color=color, lw=0.9, ls=ls, label=name)
        ax.scatter(pr[0, ::60], pr[1, ::60], s=5, color=color, zorder=3)
    idx = [int(np.argmin(np.abs(q['grid'].numpy() - t))) for t in q['doas'].numpy()]
    pr = U[:, :2].T @ (Et - mu)
    ax.scatter(pr[0, idx], pr[1, idx], s=30, marker='x', color='#CC0000', lw=1.2, zorder=4,
               label='True DoAs')
    lo, hi = np.degrees(q['grid'][0].item()), np.degrees(q['grid'][-1].item())
    ax.annotate(rf'${lo:.0f}^\circ$', (pr[0, 0], pr[1, 0]), textcoords='offset points',
                xytext=(4, 5), fontsize=5.5, color=C_TRUE)
    ax.annotate(rf'$+{hi:.0f}^\circ$', (pr[0, -1], pr[1, -1]), textcoords='offset points',
                xytext=(4, -8), fontsize=5.5, color=C_TRUE)
    ax.set_xlabel(f'Manifold PC 1 ({pct[0]:.0f}$\\%$)')
    ax.set_ylabel(f'Manifold PC 2 ({pct[1]:.0f}$\\%$)')
    ax.legend(fontsize=5.5, frameon=False, loc='best')


#*********#
#   aux   #
#*********#
def panel_aux_cut(ax, ctx, axis):
    """The null spectrum along the aux axis at the first true DoA: a 1-D cut of the joint
    landscape, showing what the nominal manifold resolves there against the corrected one."""
    q, scen = ctx['q'], SCENARIOS[ctx['scenario']]
    n = 241
    if axis == 'range':
        lo, hi = scen['range']
        x = torch.logspace(math.log10(lo), math.log10(hi), n, dtype=torch.float64)
        rr, ff = x[None], torch.full((1, n), float(q['f_true'][0]), dtype=torch.float64)
        truth, xlabel = float(q['r_true'][0]), r'Range $r$ [$\lambda/2$]'
    else:
        lo, hi = scen['freq']
        x = torch.linspace(lo, hi, n, dtype=torch.float64)
        ff, rr = x[None], torch.full((1, n), float('inf'), dtype=torch.float64)
        truth, xlabel = float(q['f_true'][0]), r'Carrier $f$'
    th = torch.full((1, n), float(q['doas'][0]), dtype=torch.float64)
    for name, nominal in (('MUSIC', True), ('CNM-MUSIC', False)):
        P = _scan(ctx['model'], q, th, rr, ff, nominal=nominal)[0].numpy()
        ax.plot(x.numpy(), P / P.max(), color=method_style(name)['color'], lw=1.1, label=name)
    ax.axvline(truth, color='#CC0000', ls='--', lw=0.8, zorder=6)
    if axis == 'range':
        # log like the sampling, but the span is under a decade, so matplotlib labels the
        # minor ticks and they collide: fixed decade-ish majors, plain numbers, no minors
        ax.set_xscale('log')
        ticks = [t for t in (5, 10, 20, 50, 100) if float(x[0]) <= t <= float(x[-1])]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Normalized Spectrum')


#***********#
#   gain    #
#***********#
def panel_gain(ax, q, k=0):
    """Per-sensor gain of the steering vector at the k-th true DoA: nominal vs true vs
    corrected (the gain half of visualize_scenario.fig_sensors)."""
    a0, at, ac, th = steering_at(q, k)
    for a, lab, c, mk in ((a0, 'Nominal', C_NOM, 'o'), (at, 'True', C_TRUE, 's'),
                          (ac, 'CNM', C_CNM, 'x')):
        ax.plot(range(len(a)), 20 * np.log10(np.abs(a)), mk, color=c, ms=3.5, mfc='none',
                lw=0, label=lab)
    ax.set_xlabel(r'Sensor $m$')
    ax.set_ylabel('Gain [dB]')


#***********#
#   phase   #
#***********#
def panel_phase(ax, q, k=0):
    """Per-sensor phase of the steering vector at the k-th true DoA: nominal vs true vs
    corrected (the phase half of visualize_scenario.fig_sensors)."""
    a0, at, ac, th = steering_at(q, k)
    for a, lab, c, mk in ((a0, 'Nominal', C_NOM, 'o'), (at, 'True', C_TRUE, 's'),
                          (ac, 'CNM', C_CNM, 'x')):
        ax.plot(range(len(a)), np.degrees(np.angle(a)), mk, color=c, ms=3.5, mfc='none',
                lw=0, label=lab)
    ax.set_xlabel(r'Sensor $m$')
    ax.set_ylabel('Phase [deg]')


#******************#
#   correction     #
#******************#
def panel_correction(ax, q):
    """The learned correction over the whole scan: |Delta_m(theta)| per sensor, as an image
    so the sensors do not overlap. The phasor panel is one column of this; here it is every
    angle, with the true DoAs marked as in the spectra panel (viridis, as the 2-D maps)."""
    a0, ac = q['a_nom'].numpy(), q['a_cnm'].numpy()
    D = np.abs(ac / np.where(np.abs(a0) < 1e-12, 1e-12, a0) - 1.0)
    grid = np.degrees(q['grid'].numpy())
    ax.imshow(D, origin='lower', aspect='auto', cmap='viridis',
              extent=(grid[0], grid[-1], -0.5, D.shape[0] - 0.5))
    for th in np.degrees(q['doas'].numpy()):
        ax.axvline(th, color='#CC0000', ls='--', lw=0.8)
    ax.set_yticks([0, D.shape[0] - 1])
    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_ylabel(r'Sensor $m$')


#****************#
#   null depth   #
#****************#
NULL_SCENES = 300                     # scenes per severity behind the null-depth panel


@torch.no_grad()
def null_cloud(model, scenario, seed, snr=10.0, T=200, d=3):
    """J at the TRUE tuples across severity, on the nominal manifold and the corrected one:
    how much of a true steering vector still lives in the noise subspace. Batched over
    scenes -- the same quantity the loss drives, read out here as a diagnostic."""
    nom, cnm = [], []
    for i, rho in enumerate(LATENT_RHOS):
        sim = build_simulator(scenario, float(rho), snr, T, d, seed + 100 * i)
        b = sim.sample(NULL_SCENES)
        z = model.encode({'X': b['X'][:, 0][:, None], 'R_hat': b['R_hat'][:, 0][:, None]})
        En = model.noise_subspace(b['R_hat'][:, 0], d)
        doas, pos = b['doas'][:, 0], b['positions']
        rr, ff = b['ranges'][:, 0], torch.ones_like(doas)
        nom.append(subspace_ratio(En, nearfield_steering_matrix(pos, doas, rr, ff)).ravel())
        cnm.append(subspace_ratio(En, model.steer(z, pos, doas, rr, ff)).ravel())
    return (np.asarray(LATENT_RHOS),
            np.stack([x.numpy() for x in nom]), np.stack([x.numpy() for x in cnm]))


def panel_nulldepth(ax, rhos, J_nom, J_cnm):
    """Median J at the truths with its interquartile band, nominal vs corrected manifold:
    the corrected manifold keeps the true tuples orthogonal to the noise subspace as the
    mismatch grows, which is exactly what the null test needs and the nominal one loses."""
    for J, name in ((J_nom, 'MUSIC'), (J_cnm, 'CNM-MUSIC')):
        c = method_style(name)['color']
        ax.fill_between(rhos, np.percentile(J, 25, axis=1), np.percentile(J, 75, axis=1),
                        color=c, alpha=0.16, lw=0)
        ax.plot(rhos, np.median(J, axis=1), color=c, ls='-', lw=1.1, label=name)
    ax.set_yscale('log')
    ax.set_xlim(rhos[0], rhos[-1])
    ax.set_xlabel(r'Severity $\rho$')
    ax.set_ylabel(r'$J$ at the Truths')


#************#
#   latent   #
#************#
@torch.no_grad()
def latent_cloud(model, scenario, seed):
    """Encoder latents of LATENT_SCENES scenes per severity, base point, and their rho."""
    zs, rhos = [], []
    for i, rho in enumerate(LATENT_RHOS):
        sim = build_simulator(scenario, float(rho), 10.0, 200, 3, seed + 100 * i)
        batch = sim.sample(LATENT_SCENES)
        z = model.encode({'X': batch['X'][:, 0][:, None], 'R_hat': batch['R_hat'][:, 0][:, None]})
        zs.append(z.reshape(z.shape[0], -1).cpu())
        rhos.append(np.full(LATENT_SCENES, rho))
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    z = torch.cat(zs).double().numpy()
    rho = np.concatenate(rhos)
    # the severity is encoded LINEARLY (ridge R^2 ~ 0.9) but not in local neighbourhoods
    # (k-NN ~ 0.2), so t-SNE/UMAP hide it: x = cross-validated linear read-out of rho,
    # y = first principal component of the part of z orthogonal to that read-out
    cv = KFold(5, shuffle=True, random_state=0)
    rho_hat = cross_val_predict(Ridge(1.0), z, rho, cv=cv)
    r2 = 1.0 - ((rho - rho_hat) ** 2).sum() / ((rho - rho.mean()) ** 2).sum()
    print(f'latent -> rho: linear read-out R^2 = {r2:.3f} ({len(rho)} scenes)')
    return rho_hat, rho


def panel_latent(ax, rho_hat, rhos):
    """What the latent says about the severity vs the truth: linear read-out per scene
    (cross-validated), identity as reference."""
    jitter = (np.random.RandomState(0).rand(len(rhos)) - 0.5) * 0.12
    ax.plot(rhos + jitter, rho_hat, '.', color='0.6', ms=2, alpha=0.45, lw=0)   # CNM family
    med = [np.median(rho_hat[rhos == r]) for r in LATENT_RHOS]
    ax.plot(LATENT_RHOS, med, 'o', color=C_CNM, ms=3.5, label='Median')
    lim = (LATENT_RHOS[0] - 0.1, LATENT_RHOS[-1] + 0.1)
    ax.plot(lim, lim, '-', color=C_TRUE, lw=0.8, label='Identity')
    ax.set_xlim(lim)
    ax.set_ylim(lim[0] - 0.4, lim[1] + 0.4)
    ax.set_xlabel(r'True Severity $\rho$')
    ax.set_ylabel(r'Latent $\hat{\rho}$')


#************#
#   figure   #
#************#
# panel geometry [inches]: one row, all boxes PANEL_H tall, the row spanning DBL_W, one
# shared legend row above the panels (as the result figures)
PANEL_H, LEFT, GAP, RIGHT, BOTTOM, HEAD = 1.45, 0.45, 0.55, 0.62, 0.42, 0.22
XLABEL_IN = 0.17                                  # x-label distance from the axis line [in]

# the spectra panel's width in the three-panel figure; every panel keeps its size whatever
# the panel count, so a fourth panel widens the CANVAS instead of shrinking the other three
W_SPEC = DBL_W - (LEFT + 2 * GAP + 2 * PANEL_H + RIGHT)

# panel name -> (draw, wide?, y-label distance from the axis line [in]); a weight above 1.0
# marks the wide spectra panel, everything else is square at PANEL_H
PANELS = {
    'spectra':  (lambda ax, c: panel_spectra(ax, c['q'], c['ssn'], styled=c['styled']), 1.44, 0.30),
    'phasors':  (lambda ax, c: panel_phasors(ax, c['q']), 1.00, 0.18),
    'portrait': (lambda ax, c: panel_portrait(ax, c['q']), 1.00, 0.28),
    'gain':     (lambda ax, c: panel_gain(ax, c['q']), 1.00, 0.28),
    'phase':    (lambda ax, c: panel_phase(ax, c['q']), 1.00, 0.28),
    'range':    (lambda ax, c: panel_aux_cut(ax, c, 'range'), 1.00, 0.30),
    'freq':     (lambda ax, c: panel_aux_cut(ax, c, 'freq'), 1.00, 0.30),
    'readout':  (lambda ax, c: panel_aux(ax, *c['clouds']['aux'], c['axis']), 1.00, 0.30),
    'correction': (lambda ax, c: panel_correction(ax, c['q']), 1.00, 0.20),
    'nulldepth': (lambda ax, c: panel_nulldepth(ax, *c['clouds']['null']), 1.00, 0.32),
    'latent':   (lambda ax, c: panel_latent(ax, *c['clouds']['latent']), 1.00, 0.28),
}
DEFAULT_PANELS = ('spectra', 'phasors', 'latent')


def compose(q, clouds, path, ssn=None, styled=False, panels=DEFAULT_PANELS, square=False,
            model=None, scenario=None, axis=None):
    draws, weights, ylabels = zip(*(PANELS[name] for name in panels))
    panel_h = PANEL_H
    widths = [PANEL_H if square or w <= 1.0 else W_SPEC for w in weights]
    W = LEFT + sum(widths) + GAP * (len(panels) - 1) + RIGHT
    H = BOTTOM + panel_h + HEAD
    ctx = dict(q=q, clouds=clouds, ssn=ssn, styled=styled, model=model,
               scenario=scenario, axis=axis)
    fig = plt.figure(figsize=(W, H))
    x = LEFT
    axes = []
    for w in widths:
        axes.append(fig.add_axes((x / W, BOTTOM / H, w / W, panel_h / H)))
        x += w + GAP
    for ax, draw in zip(axes, draws):
        draw(ax, ctx)
    for ax, w, y_in in zip(axes, widths, ylabels):
        # axis labels at the SAME distance from the axis line on every panel (matplotlib
        # pads from the tick labels, whose width differs per panel); legend in one row
        # centred above the panel (the result figures' convention)
        ax.yaxis.set_label_coords(-y_in / w, 0.5)
        ax.xaxis.set_label_coords(0.5, -XLABEL_IN / panel_h)
        handles, labels = ax.get_legend_handles_labels()
        order = sorted(range(len(labels)), key=lambda i: labels[i])     # alphabetical, as
        # one row per legend, wrapped only when the entries are wider than their own panel
        if not labels:                                # a panel with nothing to name
            continue
        ax.legend([handles[i] for i in order], [labels[i] for i in order],   # the result plots
                  loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=len(labels), fontsize=6,
                  frameon=False, handlelength=1.4, columnspacing=1.4, handletextpad=0.4,
                  borderaxespad=0.0)                  # centred, one fixed gap between entries
    fig.savefig(path + '.pdf')                        # vector only
    plt.close(fig)


#*******************#
#   joint scenes    #
#*******************#
# the aux-axis scenarios and the axis values a scene is drawn at: a range band for the near
# field, a carrier for the broadband suites (the axis the 2-D estimation happens on)
JOINT_AXES = {
    'nearfield': ('range', [('near', (5.0, 9.0)), ('mid', (15.0, 22.0)), ('far', (35.0, 49.0)),
                            ('rfull', None)]),
    'nearfield_mismatch': ('range', [('near', (5.0, 9.0)), ('mid', (15.0, 22.0)),
                                     ('far', (35.0, 49.0)), ('rfull', None)]),
    'broadband_tones': ('freq', [('f0p2', 0.2), ('f0p5', 0.5), ('f0p8', 0.8)]),
    'broadband_ofdm_carrier': ('freq', [('f0p2', 0.2), ('f0p5', 0.5), ('f0p8', 0.8),
                                       ('ffull', None)]),
}
AUX_SCENES = 1000                     # scenes behind the estimated-vs-true read-out panel
                                      # (x n_src sources -> 3000 points)


def joint_simulator(scenario, axis, value, snr, T, d, seed):
    """One scene of a joint scenario, pinned at one point of its aux axis."""
    merged = {**DEFAULT_CONFIG, **SCENARIOS[scenario]}
    cfg = sim_config(merged)
    cfg.update(n_src_range=(d, d), snr_range=(snr, snr), snapshots_range=(T, T),
               log_snapshots=False, min_sep=math.radians(8.0),
               power_imbalance_range=(0.0, 0.0), segments=1, seed=seed)
    if axis == 'range':               # a band of the range axis, or the WHOLE span when None,
        cfg['range_range'] = (tuple(SCENARIOS[scenario]['range']) if value is None
                              else tuple(value))     # every source drawing independently
    else:                             # per-source carriers sit in a narrow band around value,
        lo, hi = SCENARIOS[scenario]['freq']          # or span the WHOLE range when value is None
        cfg['freq_range'] = ((lo, hi) if value is None else
                             (max(value - 0.05, lo), min(value + 0.05, hi)))
    return NarrowbandSimulator(cfg)


#***************#
#   joint map   #
#***************#
def panel_map2d(ax, model, q, scenario, axis, nominal, n_theta=241, n_2nd=96,
                dyn_range=40.0, cmap='Greys', topo=True):
    """One (theta, aux) map, CNM-corrected or nominal, on a shared -dyn_range..0 dB scale."""
    grid = torch.linspace(float(q['grid'][0]), float(q['grid'][-1]), n_theta,
                          dtype=torch.float64)
    th = grid.repeat(n_2nd)[None]
    if axis == 'range':
        lo, hi = SCENARIOS[scenario]['range']
        second = torch.logspace(math.log10(lo), math.log10(hi), n_2nd, dtype=torch.float64)
        rr, ff = second.repeat_interleave(n_theta)[None], torch.ones_like(th)
        truth2, ylab = q['r_true'], r'$r$ [$\lambda/2$]'
    else:
        flo, fhi = SCENARIOS[scenario]['freq']
        second = torch.linspace(max(flo, 0.02), fhi, n_2nd, dtype=torch.float64)
        rr = torch.full_like(th, torch.inf)
        ff = second.repeat_interleave(n_theta)[None]
        truth2, ylab = q['f_true'], r'$f / f_c$'
    P = _scan(model, q, th, rr, ff, nominal=nominal)[0].reshape(n_2nd, n_theta)
    PdB = 10.0 * torch.log10(P.clamp_min(1e-30)).numpy()
    PdB -= PdB.max()
    ang = np.rad2deg(grid.numpy())
    sec = second.numpy()
    if topo:                          # topography: filled bands plus thin level lines
        levels = np.linspace(-dyn_range, 0.0, 17)
        im = ax.contourf(ang, sec, np.clip(PdB, -dyn_range, 0.0), levels=levels, cmap=cmap,
                         extend='min')
        ax.contour(ang, sec, np.clip(PdB, -dyn_range, 0.0), levels=levels[::4], colors='k',
                   linewidths=0.25, alpha=0.5)
        ax.set_xlim(ang[0], ang[-1]); ax.set_ylim(sec[0], sec[-1])
    else:
        im = ax.imshow(PdB, extent=(ang[0], ang[-1], sec[0], sec[-1]), origin='lower',
                       aspect='auto', cmap=cmap, vmin=-dyn_range, vmax=0.0)
    tr = np.atleast_1d(np.asarray(truth2, dtype=float))
    doas = np.rad2deg(np.atleast_1d(q['doas'].numpy()))
    ax.scatter(doas, np.resize(tr, doas.shape), marker='x', color='r', s=34, linewidths=1.4,
               label='Truth')
    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_ylabel(ylab)
    ax.grid(False)
    return im


#**************#
#   read-out   #
#**************#
@torch.no_grad()
def aux_cloud(ckpt, scenario, axis, snr, T, d, seed, scenes=AUX_SCENES):
    """Estimated vs true aux quantity over many scenes spanning the whole aux axis."""
    scen = SCENARIOS[scenario]
    fov = math.radians(scen['fov_deg'])
    scan = fov + math.radians(DEFAULT_CONFIG['scan_margin_deg'])
    est_cfg = dict(grid_size=int(round(DEFAULT_CONFIG['grid_size'] * scan / fov)),
                   theta_range=(-scan, scan), device='cpu')
    geom = ArrayGeometry.ula(scen['M'])
    est = build(f'cnm:{ckpt}', geom, est_cfg)[1]
    merged = {**DEFAULT_CONFIG, **scen}
    cfg = sim_config(merged)
    cfg.update(n_src_range=(d, d), snr_range=(snr, snr), snapshots_range=(T, T),
               log_snapshots=False, min_sep=math.radians(8.0),
               power_imbalance_range=(0.0, 0.0), segments=1, seed=seed)
    sim = NarrowbandSimulator(cfg)
    b = sim.sample(scenes)
    doas, aux = est(b['X'][:, 0], n_src=d)
    true = (b['ranges'][:, 0] if axis == 'range' else b['freqs'][:, :d].double())
    # match the estimate to the truth by angle order (both come out angle-sorted)
    order = b['doas'][:, 0].argsort(dim=-1)
    true = true.gather(1, order) if true.ndim == 2 else true
    return np.asarray(true).ravel(), np.asarray(aux).ravel()


def panel_aux(ax, true, est, axis):
    lo, hi = float(min(true.min(), est.min())), float(max(true.max(), est.max()))
    ax.plot([lo, hi], [lo, hi], color='0.5', ls='--', lw=0.8, label='Ideal')
    ax.scatter(true, est, s=3, alpha=0.35, color=C_CNM, linewidths=0, label='CNM')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    lab = r'$r$ [$\lambda/2$]' if axis == 'range' else r'$f / f_c$'
    ax.set_xlabel('True ' + lab)
    ax.set_ylabel('Estimated ' + lab)
    if axis == 'range':                      # log spacing, but labelled as plain distances
        ax.set_xscale('log'); ax.set_yscale('log')
        ticks = [v for v in (5, 10, 20, 50, 100, 200) if lo <= v <= hi]
        for a, setter in ((ax.xaxis, ax.set_xticks), (ax.yaxis, ax.set_yticks)):
            setter(ticks)
            a.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            a.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, alpha=0.2)


# widths [inches] shared by every joint composer: two maps sharing one colorbar, the gap
# before the square read-out clearing both the colorbar label and the read-out's own y label
CBAR, MAP_GAP, CBAR_PAD, PRE_AUX, RIGHT_J = 0.10, 0.16, 0.06, 0.86, 0.12
W_MAP = (DBL_W - (LEFT + MAP_GAP + CBAR_PAD + CBAR + PRE_AUX + PANEL_H + RIGHT_J)) / 2


def _maps(fig, W, H, model, q, scenario, axis, cmap, topo):
    """Nominal | CNM maps with their colorbar at the row's left; returns the x past it."""
    x, axes = LEFT, []
    for w in (W_MAP, W_MAP):
        axes.append(fig.add_axes((x / W, BOTTOM / H, w / W, PANEL_H / H)))
        x += w + MAP_GAP
    x += CBAR_PAD - MAP_GAP
    cax = fig.add_axes((x / W, BOTTOM / H, CBAR / W, PANEL_H / H))
    for ax, nominal, name in ((axes[0], True, 'Nominal'), (axes[1], False, 'CNM')):
        im = panel_map2d(ax, model, q, scenario, axis, nominal=nominal, cmap=cmap, topo=topo)
        ax.set_title(name, fontsize=7, pad=3)
    axes[1].set_ylabel('')
    axes[1].set_yticklabels([])
    fig.colorbar(im, cax=cax)
    cax.set_title('[dB]', fontsize=6, pad=3)
    return x + CBAR


def _readout(fig, W, H, x, cloud, axis):
    """The square estimated-vs-true panel at x, legend in one row above it."""
    ax = fig.add_axes((x / W, BOTTOM / H, PANEL_H / W, PANEL_H / H))
    panel_aux(ax, *cloud, axis)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=len(labels), fontsize=6, frameon=False, handlelength=1.4,
              columnspacing=1.4, handletextpad=0.4, borderaxespad=0.0)


def compose_maps(model, q, scenario, axis, path, cmap='Greys', topo=True):
    """The two maps and their colorbar alone, the maps at the full figure's size."""
    H = BOTTOM + PANEL_H + HEAD
    W = LEFT + 2 * W_MAP + MAP_GAP + CBAR_PAD + CBAR + RIGHT_J
    fig = plt.figure(figsize=(W, H))
    _maps(fig, W, H, model, q, scenario, axis, cmap, topo)
    fig.savefig(path + '.pdf')
    plt.close(fig)


def compose_readout(cloud, axis, path):
    """The read-out panel alone, as a square figure."""
    H = BOTTOM + PANEL_H + HEAD
    W = LEFT + PANEL_H + RIGHT_J
    fig = plt.figure(figsize=(W, H))
    _readout(fig, W, H, LEFT, cloud, axis)
    fig.savefig(path + '.pdf')
    plt.close(fig)


def compose_joint(model, q, cloud, scenario, axis, path, cmap='Greys', topo=True):
    """Nominal map | CNM map | estimated-vs-true read-out, on the result-figure geometry."""
    H = BOTTOM + PANEL_H + HEAD
    fig = plt.figure(figsize=(DBL_W, H))
    x = _maps(fig, DBL_W, H, model, q, scenario, axis, cmap, topo)
    _readout(fig, DBL_W, H, x + PRE_AUX, cloud, axis)
    fig.savefig(path + '.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--scenario', default='mismatch')
    parser.add_argument('--ssn', default=None, help='SubspaceNet checkpoint of the scenario')
    parser.add_argument('--out', default='results/qualitative')
    parser.add_argument('--seeds', type=int, default=40)
    parser.add_argument('--aux_scenes', type=int, default=None,
                        help='override AUX_SCENES for the read-out cloud')
    parser.add_argument('--only_seed', type=int, default=None, help='render this seed only')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--cmaps', default='Greys',
                        help='comma-separated colormaps; one file per entry when several')
    parser.add_argument('--parts', default='both', choices=['both', 'maps', 'readout'],
                        help='joint figure: maps + read-out, the maps alone, or the '
                             'read-out alone (one file per condition)')
    parser.add_argument('--only_cond', default=None, choices=['easy', 'hard'])
    parser.add_argument('--flat', action='store_true',
                        help='draw the joint maps as images instead of contour topography')
    parser.add_argument('--only_axis', default=None,
                        help='render only this axis tag (e.g. ffull, f0p5, near)')
    parser.add_argument('--square', action='store_true',
                        help='every panel as wide as it is tall')
    parser.add_argument('--panels', action='append', default=None,
                        help='comma-separated panel names (%s); repeat for several layouts, '
                             'each rendered as seed<N>__<layout>.pdf' % ', '.join(PANELS))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    use_paper_style()
    os.makedirs(args.out, exist_ok=True)
    model = CNMFramework.load_from_checkpoint(args.ckpt, map_location='cpu').eval()
    layouts = [tuple(p.split(',')) for p in args.panels] if args.panels else [DEFAULT_PANELS]
    for panels in layouts:                               # fail before the expensive clouds
        unknown = [p for p in panels if p not in PANELS]
        if unknown:
            raise SystemExit(f'unknown panel(s) {unknown}; choose from {list(PANELS)}')
    cmaps = [c.strip() for c in args.cmaps.split(',') if c.strip()]
    if args.scenario in JOINT_AXES and args.panels:       # joint scenes, single-row composer
        axis, values = JOINT_AXES[args.scenario]
        for cond, snr, T in CONDITIONS:
            clouds = {'aux': aux_cloud(args.ckpt, args.scenario, axis, snr, T, 3, seed=99,
                                       scenes=args.aux_scenes or AUX_SCENES)}
            for tag_v, value in values:
                folder = os.path.join(args.out, args.scenario, tag_v, cond)
                os.makedirs(folder, exist_ok=True)
                for seed in ([args.only_seed] if args.only_seed
                             else range(1, args.seeds + 1)):
                    sim = joint_simulator(args.scenario, axis, value, snr, T, 3, 1000 * seed)
                    q = scene_quantities(model, sim.sample(1), 0)
                    for panels in layouts:
                        tag = '' if len(layouts) == 1 else '__' + '-'.join(panels)
                        compose(q, clouds, os.path.join(folder, f'seed{seed}{tag}'),
                                panels=panels, square=args.square, model=model,
                                scenario=args.scenario, axis=axis)
                print(folder, f'{len(layouts)} layout(s)')
        print('figures ->', args.out)
        return
    if args.scenario in JOINT_AXES:                      # 2-D estimation figure
        axis, values = JOINT_AXES[args.scenario]
        for cond, snr, T in CONDITIONS:
            if args.only_cond and cond != args.only_cond:
                continue
            # the read-out cloud is the expensive step and the maps never use it
            cloud = (None if args.parts == 'maps' else
                     aux_cloud(args.ckpt, args.scenario, axis, snr, T, 3, seed=99,
                               scenes=args.aux_scenes or AUX_SCENES))
            if args.parts == 'readout':               # independent of tag and seed
                folder = os.path.join(args.out, args.scenario)
                os.makedirs(folder, exist_ok=True)
                compose_readout(cloud, axis, os.path.join(folder, f'readout_{cond}'))
                print(folder, f'readout_{cond}')
                continue
            for tag, value in values:
                if args.only_axis and tag != args.only_axis:
                    continue
                folder = os.path.join(args.out, args.scenario, tag, cond)
                os.makedirs(folder, exist_ok=True)
                for seed in ([args.only_seed] if args.only_seed
                             else range(1, args.seeds + 1)):
                    sim = joint_simulator(args.scenario, axis, value, snr, T, 3, 1000 * seed)
                    q = scene_quantities(model, sim.sample(1), 0)
                    for cmap in cmaps:
                        suffix = '' if len(cmaps) == 1 else f'__{cmap}'
                        target = os.path.join(folder, f'seed{seed}{suffix}')
                        if args.parts == 'maps':
                            compose_maps(model, q, args.scenario, axis, target, cmap=cmap,
                                         topo=not args.flat)
                        else:
                            compose_joint(model, q, cloud, args.scenario, axis, target,
                                          cmap=cmap, topo=not args.flat)
                print(folder, f'{1 if args.only_seed else args.seeds} seed(s)',
                      f'{len(cmaps)} cmap(s)')
        print('figures ->', args.out)
        return
    clouds = {'latent': latent_cloud(model, args.scenario, seed=99)}
    if any('nulldepth' in panels for panels in layouts):
        clouds['null'] = null_cloud(model, args.scenario, seed=99)
    ssn_est = None
    if args.ssn:
        scen = SCENARIOS[args.scenario]
        fov = math.radians(scen['fov_deg'])
        scan = fov + math.radians(DEFAULT_CONFIG['scan_margin_deg'])
        ssn_est = SubspaceNetEstimator(args.ssn, ArrayGeometry.ula(scen['M']),
                                       dict(grid_size=int(round(DEFAULT_CONFIG['grid_size']
                                                                * scan / fov)),
                                            theta_range=(-scan, scan), device='cpu'),
                                       method='music')
    for seed in ([args.only_seed] if args.only_seed else range(1, args.seeds + 1)):
        for rho in RHOS:
            for cond, snr, T in CONDITIONS:
                sim = build_simulator(args.scenario, rho, snr, T, 3, 1000 * seed)
                batch = sim.sample(1)
                q = scene_quantities(model, batch, 0)
                ssn = ssn_spectrum(ssn_est, batch['X'][0, 0], q['d']) if ssn_est else None
                folder = os.path.join(args.out, args.scenario, f'rho{rho:.1f}'.replace('.', 'p'),
                                      cond)
                os.makedirs(folder, exist_ok=True)
                for panels in layouts:
                    tag = '' if len(layouts) == 1 else '__' + '-'.join(panels)
                    compose(q, clouds, os.path.join(folder, f'seed{seed}{tag}'), ssn=ssn,
                            panels=panels, square=args.square, model=model,
                            scenario=args.scenario)
                print(folder, f'seed{seed}', f'{len(layouts)} layout(s)')
    print('figures ->', args.out)


if __name__ == '__main__':
    main()
