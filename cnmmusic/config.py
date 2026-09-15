####################################################################################################
#                                            config.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: The single default configuration and the dict -> argparse -> wandb merge helpers.       #
#          Ranges are (lo, hi) uniform draws per condition; scalars are fixed. Training is         #
#          steps-based (-1 = run until stopped).                                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
import torch


#***************************#
#   default configuration   #
#***************************#
DEFAULT_CONFIG = {
    # model
    'model': 'cnm',                        # model family: 'cnm' | 'subspacenet' | 'damusic'
                                           # | 'gridcnn'
    'encoder': 'snapattn',                 # observation encoder: 'snapshot' (GRU over raw X)
                                           # | 'covariance' (triu features of R) | 'lagcov'
                                           # (tau-lag features) | 'snapattn' (attention
                                           # tokenizer over raw snapshots) | 'meanset'
                                           # (masked mean over encoded snapshots)
    'z_dim': 32,                           # latent width of the mlp/sphere backend
    'hidden': 128,
    'backend': 'sphere',                   # latent path after the encoder: 'none' | 'mlp' |
                                           # 'sphere' (unit-norm latent, enables vMF sampling)

    'train_cov': 'empirical',              # covariance-encoder training input: 'empirical'
                                           # (R_hat) | 'true' (exact covariance)
    'train_En': 'empirical',               # noise subspace in the loss: 'empirical' | 'true';
                                           # validation always uses the sample estimate

    'val_estimators': ['music'],           # validation back-ends, each with its nominal
                                           # 0-reference: 'music' | 'mvdr' | 'root' | 'sbl'
                                           # | 'bbmusic' | 'swmusic' | 'gevd'
    'loss': 'nlls',                        # training loss: 'ce' (one-hot CE over truth vs
                                           # negatives) | 'ce-beam' (CE with soft targets =
                                           # the nominal beampattern overlap; no radius or
                                           # mask parameters) | 'rank' (hinge on log J) |
                                           # 'nll' (CE with exclusion mask) | 'nlls' (CE with
                                           # soft targets from the manifold deficit)
    'tau_nll': 1.0,                        # softmax temperature on -log J
    'eps_scale': 1.0,                      # width of the soft-target bump (loss 'nlls') in units
                                           # of the min_sep/2 manifold ring: 1.0 = the ring,
                                           # smaller = a bump closer to the data resolution
    'pin_aux': False,                      # pin range and frequency at nominal in the loss and
                                           # scan: theta-only supervision, the correction absorbs
                                           # the aux axes as mismatch
    'corr_mode': 'mult',                   # corrected manifold: 'mult' (a0 * (1 + Delta)) |
                                           # 'add' (a0 + Delta) | 'free' (Delta alone)
    'corr_cond': 'concat',                 # scene conditioning: 'concat' (z into the first
                                           # layer) | 'film' (z -> per-layer scale/shift) |
                                           # 'attn' (cross-attention over encoder tokens)
    'coord_enc': 'harm',                   # coordinate encoding: 'harm' (fixed sin/cos comb)
                                           # | 'lff' (learnable frequencies) | 'raw'
                                           # (standardized scalars)
    'n_neg': 2048,                         # random negative tuples per scene
    'margin': 2.0,                         # hinge margin (loss 'rank')
    'excl': None,                          # candidate exclusion around truths (nll/rank
                                           # only): None (manifold metric, auto eps) | float
                                           # (span-normalized parameter-space radius)
    'n_hard': 0,                           # hard negatives mined per scene (0 = off)
    'n_probe': 256,                        # probes drawn per scene when n_hard > 0
    'n_harm': 16,                          # sin/cos harmonics per coordinate axis
    'tau': 8,                              # lags for the lagcov encoder (and subspacenet)

    # estimation difficulty
    'snr': (-10.0, 30.0),                  # dB
    'snapshots': (1, 1000),                # T, uniform int per batch
    'n_src': (1, 7),                       # sources d, uniform int per batch (max M-1)
    'min_sep_deg': 2.0,                    # min angular separation
    'power_imbalance_db': (0.0, 6.0),      # per-source power spread

    # scan margin beyond the FOV [deg]: truths stay inside the FOV, but every scanned
    # theta grid extends past it so edge nulls are whole and edge peaks are interior
    'scan_margin_deg': 5.0,

    # geometry and aperture
    'M': 8,                                # sensors
    'geometries': ['ula'],                 # training mixture: 'ula' | 'uca' | 'ura' | 'nula'
                                           # | 'random_planar'
    'spacing': (1.0, 1.0),                 # adjacent-element spacing [half-wavelengths]
    'fov_deg': {'ula': 60.0, 'nula': 60.0,             # +- half FOV per geometry
                'uca': 180.0, 'ura': 180.0, 'random_planar': 180.0},

    # source correlation (1 = fully coherent)
    'source_corr': (0.0, 1.0),

    # spatially colored noise: AR-1 coefficient of Q_ij = nc^|i-j|; None = white
    'noise_corr': None,

    # distributed sources: per-source angular spread [deg]; None = point sources
    'spread_deg': None,
    'n_failed': None,
    # moving sources: per-source total window drift [deg]; None = static
    'motion_deg': None,

    # mismatch / calibration: rho (float) = canonical imperfection pattern scaled by rho,
    # bounds below ignored; rho None = per-condition severities ~ U(lo, hi) per type
    'rho': None,
    'imperfections': {
        'gain': (0.0, 0.3),                # amplitude error fraction
        'phase_deg': (0.0, 45.0),          # phase error
        'pos': (0.0, 0.3),                 # position error (fraction of spacing)
        'coupling_mag': (0.0, 0.45),       # Toeplitz coupling |gamma|
        'coupling_phase_deg': (0.0, 360.0),
    },

    # source distance [half-wavelengths]: None = far field; (lo, hi) = uniform draw
    'range': None,

    # band
    'narrowband': True,
    'freq': (1.0, 1.0),                    # source frequency f/fc

    # training (steps-based)
    'max_steps': -1,                       # -1 = run until stopped
    'val_every_steps': 500,
    'batch': 64,
    'trueBatch': 64,
    'lr': 1e-3,
    'weight_decay': 0.0,                   # Adam L2 (baselines; SubspaceNet ref 1e-9)
    'optimizer': 'adam',                   # 'adam' | 'adamw'
    'segments': 1,                         # recordings per condition
    'grid_size': 480,                      # shared grid resolution for every scanned axis
    'callback': 'val/music/doa_rmspe_deg', # checkpoint monitor metric
    'num_workers': 0,

    # infra
    'online': False,
    'entity': 'jume',                      # wandb entity (explicit: the account default is
                                           # a different team entity)
    'gpu_selection': 0,
    'load_model': '',                      # checkpoint path to warm-start from
    'resume': '',                          # wandb run id to resume: same row, last.ckpt
                                           # auto-located, stream and RNG states restored
    'seed': 42,
}


#*******************#
#   merge helpers   #
#*******************#
def parse_cli(defaults=None):
    """Dynamically expose every scalar default as a CLI flag (dict -> argparse)."""
    defaults = defaults if defaults is not None else DEFAULT_CONFIG
    parser = argparse.ArgumentParser()
    for key, val in defaults.items():
        if val is None or isinstance(val, (list, tuple, dict, torch.device)):
            continue
        parser.add_argument('--' + key, default=val,
                            type=(lambda s: s.lower() in ('1', 'true', 'yes'))
                            if isinstance(val, bool) else type(val))
    args, _ = parser.parse_known_args()
    merged = dict(defaults)
    merged.update(vars(args))
    return merged


def sim_config(config):
    """Slice the merged config into the simulator's config dict."""
    from cnmmusic.arrays.imperfections import ImperfectionModel
    rho, rho_range = config.get('rho'), None
    if rho is not None:
        if isinstance(rho, (tuple, list)):     # severity range: per-condition draw in the sim
            imp = ImperfectionModel(rho=1.0, seed=config['seed'])
            rho_range = tuple(rho)
        else:
            imp = ImperfectionModel(rho=float(rho), seed=config['seed'])
    elif config['imperfections']:
        imp = ImperfectionModel(bounds=config['imperfections'], seed=config['seed'])
    else:
        imp = None
    return {
        'rho_range': rho_range,
        'M': config['M'], 'imperfections': imp,
        'geometries': config['geometries'], 'spacing': config['spacing'],
        'freeze_geometry': config.get('freeze_geometry', False),
        'fov_deg': config['fov_deg'],
        'n_src_range': tuple(config['n_src']),
        'snr_range': tuple(config['snr']),
        'snapshots_range': tuple(config['snapshots']), 'log_snapshots': False,
        'min_sep': float(torch.deg2rad(torch.tensor(config['min_sep_deg']))),
        'source_corr_range': tuple(config['source_corr']),
        'noise_corr_range': tuple(config['noise_corr']) if config.get('noise_corr') else None,
        'spread_deg': tuple(config['spread_deg']) if config.get('spread_deg') else None,
        'motion_deg': tuple(config['motion_deg']) if config.get('motion_deg') else None,
        'n_failed': tuple(config['n_failed']) if config.get('n_failed') else None,
        'power_imbalance_range': tuple(config['power_imbalance_db']),
        'range_range': tuple(config['range']) if config['range'] else None,
        'freq_range': tuple(config['freq']),
        'per_source_freq': config.get('per_source_freq', False),
        'signal': config.get('signal', 'gauss'),
        'subcarriers': int(config.get('subcarriers') or 0),
        'bandwidth': float(config.get('bandwidth') or 0.0),
        'coarray': config.get('coarray', False),
        'segments': config['segments'], 'grid_size': config['grid_size'],
        'seed': config['seed'],
    }
