####################################################################################################
#                                           visualize.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Publication-quality visualization utilities (spectra, array geometries, eigenvalues,    #
#          near-field maps) using the scienceplots science + ieee styles (LaTeX text rendering,    #
#          IEEE column widths); dB spectra capped to a fixed dynamic range, red dashed true-DoA    #
#          markers, colorblind-safe lines.                                                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import torch

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the 'science' styles)


#*****************#
#   paper style   #
#*****************#
COL_W = 3.45                                       # IEEE single-column width [in]
DBL_W = 7.16                                       # IEEE double-column width [in]

# one fixed style per method across ALL figures (Okabe-Ito colorblind-safe palette);
# line style encodes the family: ':' classical, '--' data-driven, '-.' CNM, bounds unmarked
METHOD_STYLES = {
    'MUSIC':          dict(color='#0173B2', ls=':',  marker='^'),
    'Root-MUSIC':     dict(color='#DE8F05', ls=':',  marker='v'),
    'ESPRIT':         dict(color='#029E73', ls=':',  marker='s'),
    'MVDR':           dict(color='#CC78BC', ls=':',  marker='D'),
    'SS-MUSIC':       dict(color='#946635', ls=':',  marker='p'),
    'MLE':            dict(color='#999933', ls=':',  marker='*'),
    'MUSIC (2D)':     dict(color='#0077BB', ls=':',  marker='<'),
    'MUSIC (Cascade)': dict(color='#AA4499', ls=':', marker='P'),
    'MVDR (2D)':      dict(color='#CC78BC', ls=':',  marker='>'),
    'MVDR (Cascade)': dict(color='#EE3377', ls=':',  marker='P'),
    'ESPRIT (JAFE)':  dict(color='#029E73', ls=':',  marker='s'),
    'NF-MLE':         dict(color='#009988', ls=':',  marker='X'),
    'DA-MUSIC':       dict(color='#56B4E9', ls='--', marker='o'),
    'SubspaceNet':    dict(color='#D55E00', ls='--', marker='h'),
    'SubspaceNet-MUSIC': dict(color='#332288', ls='--', marker='>'),
    'GridCNN':        dict(color='#AA3377', ls='--', marker='X'),
    'CNM-MUSIC':      dict(color='k',       ls='-.', marker='o'),
    'CNM-DA-MUSIC':   dict(color='0.35',    ls='-.', marker='s'),
    'CNM-MVDR':       dict(color='0.55',    ls='-.', marker='d'),
    'CNM-Root-MUSIC': dict(color='k',       ls='-.', marker='o'),
    'CNM-SubspaceNet-MUSIC': dict(color='#332288', ls='-', marker='>'),
    'CNM-SubspaceNet-MVDR':  dict(color='#332288', ls=(0, (3, 1, 1, 1)), marker='<'),
    'CNM-ESPRIT':     dict(color='0.55',    ls='-.', marker='d'),
    'Oracle-Root':    dict(color='0.45',    ls=(0, (5, 2)), marker=''),
    'Oracle':         dict(color='0.65',    ls=(0, (5, 2)), marker=''),
    'CRB':            dict(color='k',       ls=':',  marker=''),
    'ZZB':            dict(color='0.45',    ls=(0, (4, 1.5)), marker=''),
}


def method_style(name, i=0):
    return METHOD_STYLES.get(name, dict(color=f'C{i}', ls='-', marker='o'))


AXIS_LABELS = {
    'snr': 'SNR [dB]',
    'snapshots': r'Snapshots $T$',
    'n_src': r'Number of Sources $D$',
    'delta': r'Source Separation $\Delta\theta$ [deg]',
    'freq': r'Carrier Offset [$\%$ of $\lambda/2$ spacing]',
    'range': r'Nearfield Range $r$ [$\lambda/2$]',
    'rho': r'Mismatch Severity $\rho$',
    'source_corr': r'Source Correlation',
    'noise_corr': r'Noise Correlation',
    'M': r'Number of Sensors $M$',
    'subcarriers': r'Number of Subcarriers',
    'f_center': r'Carrier Frequency $f / f_c$',
    'spread_deg': r'Angular Spread [deg]',
    'n_failed': r'Failed Sensors',
}


def use_paper_style(latex=None):
    """
    Apply the science/ieee style globally (call once per process before plotting).
    latex=None auto-detects a LaTeX installation and falls back to mathtext without one.
    """
    if latex is None:
        import shutil
        latex = shutil.which('latex') is not None
    styles = ['science', 'ieee'] if latex else ['science', 'ieee', 'no-latex']
    plt.style.use(styles)
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'axes.grid': True,
        'grid.alpha': 0.15,
        'legend.frameon': False,
        'lines.linewidth': 1.0,
    })


def save_fig(fig, path):
    """Save both the vector PDF (paper) and a PNG preview next to it."""
    import pathlib
    path = pathlib.Path(path)
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'))


#*************#
#   helpers   #
#*************#
def to_dB(P):
    return 10.0 * np.log10(np.abs(np.asarray(P, dtype=np.float64)) + 1e-30)


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


#*********************#
#   angular spectra   #
#*********************#
def plot_spectra_overlay(ax, grid, spectra, labels, title=None, true_doas=None):
    """
    Peak-normalized overlay of several methods' (pseudo-)spectra on a linear scale (a MUSIC
    pseudo-spectrum is not a power quantity, so no dB axis); fixed method colors; the true
    DoAs are broad light bands drawn BEHIND all curves so they never cover an estimate.
    """
    ang = np.rad2deg(_np(grid))
    if true_doas is not None:                       # red dashed ON TOP (dashes keep curves visible)
        for k, a in enumerate(np.atleast_1d(np.rad2deg(_np(true_doas)))):
            ax.axvline(a, color='#CC0000', ls='--', lw=1.0, zorder=4,
                       label='true' if k == 0 else None)
    for i, (P, lab) in enumerate(zip(spectra, labels)):
        P = _np(P).astype(np.float64)
        st = method_style(lab, i)
        ax.plot(ang, P / P.max(), linewidth=1.0, color=st['color'], ls='-', label=lab)
    ax.set_xlabel(r'$\theta$ [deg]'); ax.set_ylabel('Normalized spectrum')
    if title:
        ax.set_title(title)
    ax.set_xlim(ang[0], ang[-1])
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02, 1.0, 0.2), mode='expand',
              ncol=min(len(labels) + (true_doas is not None), 4), fontsize=5.5,
              handlelength=1.4, columnspacing=0.8, borderaxespad=0.0, frameon=False)
    ax.grid(True, alpha=0.15)


#*******************************#
#   near-field (theta, r) map   #
#*******************************#
def plot_nearfield_spectrum(ax, grid, ranges, P, title=None, true_doas=None,
                            true_ranges=None, dyn_range=40.0):
    """2-D pseudo-spectrum over angle x range: P (R, G) -> dB image with truth markers."""
    PdB = to_dB(_np(P))
    PdB = PdB - PdB.max()
    ang = np.rad2deg(_np(grid))
    rng = _np(ranges)
    im = ax.imshow(PdB, extent=(ang[0], ang[-1], rng[0], rng[-1]), origin='lower',
                   aspect='auto', cmap='viridis', vmin=-dyn_range, vmax=0.0)
    if true_doas is not None and true_ranges is not None:
        ax.scatter(np.rad2deg(_np(true_doas)), _np(true_ranges), marker='x', color='r', s=60,
                   linewidths=2)
    ax.set_xlabel(r'$\theta$ [deg]'); ax.set_ylabel(r'$r$ [$\lambda/2$]')
    if title:
        ax.set_title(title)
    ax.grid(False)
    return im


#****************************#
#   (theta, f) carrier map   #
#****************************#
def plot_freq_spectrum(ax, grid, f_grid, P, title=None, true_doas=None, true_f=None,
                       dyn_range=40.0):
    """2-D pseudo-spectrum over angle x carrier offset: P (F, G) -> dB image with truth marks."""
    PdB = to_dB(_np(P))
    PdB = PdB - PdB.max()
    ang = np.rad2deg(_np(grid))
    off = 100.0 * (_np(f_grid) - 1.0)                       # carrier offset [% of f_c]
    im = ax.imshow(PdB, extent=(ang[0], ang[-1], off[0], off[-1]), origin='lower',
                   aspect='auto', cmap='viridis', vmin=-dyn_range, vmax=0.0)
    if true_doas is not None and true_f is not None:
        f_off = 100.0 * (float(true_f) - 1.0)
        ax.scatter(np.rad2deg(_np(true_doas)), np.full(np.atleast_1d(_np(true_doas)).shape,
                                                       f_off),
                   marker='x', color='r', s=60, linewidths=2)
    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_ylabel(r'Carrier Offset [$\%$ of $f_c$]')
    if title:
        ax.set_title(title)
    ax.grid(False)
    return im


#**********************#
#   array geometries   #
#**********************#
def plot_array(ax, positions, title=None, doas=None):
    """Sensor positions (half-wavelength units) with optional DoA arrows from far field."""
    p = _np(positions)
    ax.scatter(p[:, 0], p[:, 1], s=50, zorder=3)
    for n, (x, y) in enumerate(p[:, :2]):
        ax.annotate(f'$p_{{{n}}}$', (x, y), textcoords='offset points', xytext=(4, 6), fontsize=8)
    if doas is not None:
        span = max(np.ptp(p[:, 0]), 1.0)
        for a in np.atleast_1d(_np(doas)):
            u = np.array([np.sin(a), np.cos(a)])
            start = u * span * 1.2
            ax.annotate('', xy=(0, 0), xytext=start,
                        arrowprops=dict(arrowstyle='->', color='r', alpha=0.7))
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$ [$\lambda/2$]'); ax.set_ylabel(r'$y$ [$\lambda/2$]')
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.15)


#*****************#
#   eigenvalues   #
#*****************#
def plot_eigenvalues(ax, R, title=None):
    """Descending eigenvalue stem plot of a covariance (source-count sanity check)."""
    evals = np.sort(np.linalg.eigvalsh(_np(R)))[::-1]
    ax.stem(np.arange(1, len(evals) + 1), evals)
    ax.set_yscale('log')
    ax.set_xlabel('Index'); ax.set_ylabel('Eigenvalue')
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.15)


#****************#
#   root plots   #
#****************#
def plot_roots(ax, roots, selected=None, title=None, true_doas=None, spacing=1.0):
    """
    Root-MUSIC roots on the complex plane with the unit circle (subspace-methods style).
    roots: complex (n,); selected: indices of the chosen (source) roots; true_doas marks the
    TRUE angles' positions on the circle (z = e^{-j pi spacing sin(theta)}).
    """
    t = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(t), np.sin(t), color='0.6', lw=0.7)
    r = _np(roots)
    ax.scatter(r.real, r.imag, s=14, facecolors='none', edgecolors='C0', lw=0.8,
               label='roots')
    if selected is not None:
        ax.scatter(r[selected].real, r[selected].imag, s=22, color='r', marker='x', lw=1.2,
                   label='selected')
    if true_doas is not None:
        z = np.exp(-1j * np.pi * spacing * np.sin(_np(true_doas)))
        ax.scatter(z.real, z.imag, s=34, color='#029E73', marker='+', lw=1.2, label='true')
    ax.set_aspect('equal')
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)  # roots far off the circle are irrelevant
    ax.set_xlabel(r'$\Re$'); ax.set_ylabel(r'$\Im$')
    if title:
        ax.set_title(title)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02, 1.0, 0.2), mode='expand',
              ncol=3, fontsize=5.5, handlelength=1.0, columnspacing=1.2,
              borderaxespad=0.0, frameon=False)
    ax.grid(True, alpha=0.15)
