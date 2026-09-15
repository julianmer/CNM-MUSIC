####################################################################################################
#                                           scenarios.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: The scenarios (pure data domains), designed as one-axis-at-a-time probes: BASE fixes    #
#          a single operating point (10 dB SNR, 200 snapshots, 3 sources, calibrated far-field     #
#          ULA at the design frequency), and every other scenario re-opens exactly ONE             #
#          difficulty axis around it. GEOMETRY is orthogonal to the scenario: pass                 #
#          --geometry ula|uca|ura|nula|random_planar to train_scenario.py.                         #
#                                                                                                  #
####################################################################################################


#***************#
#   scenarios   #
#***************#
# fixed operating point: every axis pinned -- plane waves, calibrated array, white noise,
# uncorrelated equal-power sources
BASE = {
    'M': 8,
    'geometries': ['ula'],
    'fov_deg': 60.0,                # +- half-FOV, same for every geometry
    'spacing': (1.0, 1.0),
    'freq': (1.0, 1.0),
    'snr': (10.0, 10.0),
    'snapshots': (200, 200),
    'n_src': (3, 3),
    'min_sep_deg': 2.0,
    'power_imbalance_db': (0.0, 0.0),
    'source_corr': (0.0, 0.0),
    'imperfections': None,
    'rho': None,
    'range': None,                  # exact far field (plane waves, u = 1/r = 0)
}

# estimation-difficulty axes
SNR = {**BASE, 'snr': (-10.0, 30.0)}
SNAPSHOTS = {**BASE, 'snapshots': (1, 400)}
SOURCES = {**BASE, 'n_src': (1, 5)}

# two sources at the base point: the close-pair resolution axis (exact separation at eval);
# separations keep the shared 2 deg floor -- below that the pair merges and the CRB
# turns unstable
SEPARATION = {**BASE, 'n_src': (2, 2)}

# pairwise source correlation up to full coherence (corr = 1 -> rank-1 source covariance)
CORRELATED = {**BASE, 'source_corr': (0.0, 1.0)}

# spatially colored noise of unknown covariance (AR-1 Toeplitz Q_ij = nc^|i-j|); known-Q
# GEVD is the oracle baseline
COLORED_NOISE = {**BASE, 'noise_corr': (0.0, 0.99)}

# calibration errors as one severity scalar: the canonical imperfection pattern scaled by rho
MISMATCH = {**BASE, 'rho': (0.0, 1.5)}

# source distance inside the measurable-curvature regime (M = 8 ULA: Fresnel edge ~8,
# Fraunhofer ~49 half-wavelengths)
NEARFIELD = {**BASE, 'range': (5.0, 49.0)}

# near field under calibration errors: the curvature signature is second-order phase, so
# mismatch buries the range information long before it noticeably harms far-field angles
NEARFIELD_MISMATCH = {**NEARFIELD, 'rho': (0.0, 1.5)}

# distributed sources: per-source angular spread, coherent rays -> effective steering;
# centers inside +-55 deg so the widest spread's rays stay within the +-60 deg field of view
DISTRIBUTED = {**BASE, 'spread_deg': (0.0, 10.0), 'fov_deg': 55.0}

# moving sources: linear per-source drift across the window, truth at the window center
MOVING = {**BASE, 'motion_deg': (0.0, 10.0)}

# per-source tone carriers over the whole band, normalized to the design f_max
# (spacing = lambda_min/2, band = (0, 1]); line spectra with per-subcarrier steering
BROADBAND_TONES = {**BASE, 'signal': 'tone', 'freq': (0.0, 1.0), 'per_source_freq': True}

# baseband OFDM occupies the whole band -- no per-source carrier exists, so the model's
# f axis stays pinned; the band lives inside the simulator only
BROADBAND_OFDM = {**BASE, 'signal': 'ofdm', 'subcarriers': 1000, 'bandwidth': 1.0,
                  'freq': (1.0, 1.0)}

# narrow per-source OFDM around a per-source carrier
BROADBAND_OFDM_CARRIER = {**BASE, 'signal': 'ofdm', 'subcarriers': 10, 'bandwidth': 0.1,
                          'freq': (0.0, 0.9), 'per_source_freq': True}

# more sources than sensors: the nested array's contiguous coarray makes d = 12 > M = 8
# identifiable for uncorrelated sources; the simulator's coarray front-end turns every
# scene into a virtual 20-element one, and calibration errors (the swept axis) corrupt the
# PHYSICAL array before the lag map
NESTED = {**BASE, 'geometries': ['nested'], 'coarray': True, 'M_virtual': 20,
          'n_src': (12, 12)}

# the coarray under calibration errors: lag averaging assumes exact redundancy, so
# physical gain/phase/position errors corrupt the virtual array coherently
NESTED_MISMATCH = {**NESTED, 'rho': (0.0, 1.5)}

# the source count as the open axis, up to the coarray's cap (virtual ULA of 20)
NESTED_SOURCES = {**NESTED, 'n_src': (8, 19)}

# sensor failures: 0..4 of the 8 elements dead (noise only) at random positions, unknown to
# every estimator -- the nominal manifold still assumes a full array
SENSOR_FAILURE = {**BASE, 'n_failed': (0, 4)}

# unknown source count
UNKNOWN_SOURCES = {**BASE, 'known_n_src': False}

SCENARIOS = {'base': BASE,
             'snr': SNR,
             'snapshots': SNAPSHOTS,
             'sources': SOURCES,
             'separation': SEPARATION,
             'correlated': CORRELATED,
             'colored_noise': COLORED_NOISE,
             'mismatch': MISMATCH,
             'nearfield': NEARFIELD,
             'nearfield_mismatch': NEARFIELD_MISMATCH,
             'distributed': DISTRIBUTED,
             'moving': MOVING,
             'broadband_tones': BROADBAND_TONES,
             'broadband_ofdm': BROADBAND_OFDM,
             'broadband_ofdm_carrier': BROADBAND_OFDM_CARRIER,
             'nested': NESTED,
             'nested_mismatch': NESTED_MISMATCH,
             'nested_sources': NESTED_SOURCES,
             'sensor_failure': SENSOR_FAILURE,
             'unknown_sources': UNKNOWN_SOURCES}
