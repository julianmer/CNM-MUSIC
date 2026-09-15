####################################################################################################
#                                        test_estimators.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: End-to-end sanity tests: simulator -> classical estimators -> RMSPE.                    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch
import pytest

from cnmmusic.arrays.geometry import ArrayGeometry
from cnmmusic.arrays.imperfections import ImperfectionModel
from cnmmusic.data.simulator import NarrowbandSimulator
from cnmmusic.estimators.classical import (MUSIC, RootMUSIC, ESPRIT, MVDR, OracleMUSIC,
                                           SpatialSmoothingMUSIC)
from cnmmusic.criteria.metrics import rmspe, wrapped_diff


#*************#
#   helpers   #
#*************#
def make_batch(**overrides):
    # +-60 deg: the standard ULA field of view (endfire excluded for stability)
    cfg = {'snr_range': (10.0, 10.0), 'snapshots_range': (100, 100), 'seed': 7,
           'theta_range': (-1.05, 1.05), 'min_sep': 0.25}    # 14 deg: above MVDR's beamwidth
    cfg.update(overrides)
    sim = NarrowbandSimulator(cfg)
    return sim, sim.sample(16)


def run_estimator(est, batch, **kw):
    preds, trues = [], []
    for b in range(batch['X'].shape[0]):
        preds.append(est(batch['X'][b, 0], n_src=int(batch['n_src'][0]), **kw))
        trues.append(batch['doas'][b, 0])
    return torch.stack(preds), torch.stack(trues)


#************************#
#   ideal-array sanity   #
#************************#
@pytest.mark.parametrize('est_cls', [MUSIC, RootMUSIC, ESPRIT, MVDR])
def test_ideal_array_accuracy(est_cls):
    sim, batch = make_batch()
    est = est_cls(ArrayGeometry.ula(8))
    pred, true = run_estimator(est, batch)
    err = torch.rad2deg(rmspe(pred, true)).item()
    assert err < 2.0, f'{est_cls.__name__}: RMSPE {err:.2f} deg on an ideal array at 10 dB'


def test_music_beats_random():
    sim, batch = make_batch(snr_range=(-5.0, -5.0))
    est = MUSIC(ArrayGeometry.ula(8))
    pred, true = run_estimator(est, batch)
    err = torch.rad2deg(rmspe(pred, true)).item()
    assert err < 20.0


#*****************************#
#   mismatch and oracle gap   #
#*****************************#
def test_mismatch_degrades_and_oracle_recovers():
    imp = ImperfectionModel(rho=1.0, randomized=True, seed=11)
    sim, batch = make_batch(imperfections=imp)
    geom = ArrayGeometry.ula(8)

    nominal = MUSIC(geom, config={'grid_size': sim.grid_size,
                                  'theta_range': sim.theta_range})
    oracle = OracleMUSIC(geom, config={'grid_size': sim.grid_size,
                                       'theta_range': sim.theta_range})
    pred_n, true = run_estimator(nominal, batch)
    pred_o = torch.stack([oracle(batch['X'][b, 0], n_src=int(batch['n_src'][0]),
                                 A_true=batch['a_true_grid'][b])
                          for b in range(batch['X'].shape[0])])
    err_n = torch.rad2deg(rmspe(pred_n, true)).item()
    err_o = torch.rad2deg(rmspe(pred_o, true)).item()
    assert err_o < err_n, f'oracle ({err_o:.2f}) should beat nominal ({err_n:.2f}) under mismatch'
    assert err_o < 2.0, f'oracle should be accurate, got {err_o:.2f} deg'


#**********************#
#   coherent sources   #
#**********************#
def test_spatial_smoothing_on_coherent():
    # theta restricted away from endfire: the smoothed subarray aperture cannot resolve there
    sim, batch = make_batch(source_corr=1.0, min_sep=0.35, theta_range=(-1.05, 1.05))
    geom = ArrayGeometry.ula(8)
    plain = MUSIC(geom)
    smooth = SpatialSmoothingMUSIC(geom, subarray=4)
    pred_p, true = run_estimator(plain, batch)
    pred_s, _ = run_estimator(smooth, batch)
    err_p = torch.rad2deg(rmspe(pred_p, true)).item()
    err_s = torch.rad2deg(rmspe(pred_s, true)).item()
    assert err_s < 5.0, f'smoothed MUSIC should handle coherent sources, got {err_s:.2f} deg'
    assert err_s < err_p


#*************#
#   metrics   #
#*************#
def test_rmspe_permutation_invariant():
    a = torch.tensor([[0.1, -0.4]], dtype=torch.float64)
    b = torch.tensor([[-0.4, 0.1]], dtype=torch.float64)
    assert rmspe(a, b).item() < 1e-12


def test_matched_tuple_rmse_joint_assignment():
    """Near-tie in angle: the joint span-normalized cost must let the aux axis break the tie
    (angle-only matching would flip the pairing and report a catastrophic range error)."""
    from cnmmusic.criteria.metrics import matched_tuple_rmse
    deg = torch.pi / 180.0
    true_th = torch.tensor([[10.0, 12.0]]) * deg
    pred_th = torch.tensor([[11.0, 11.1]]) * deg
    r_true = torch.tensor([[10.0, 80.0]])
    r_pred = torch.tensor([[75.0, 13.0]])                   # pred0 is truth1's range partner
    u_max = 0.2
    th_err, (r_err,) = matched_tuple_rmse(
        pred_th, true_th, theta_span=2.1,
        aux=[(1.0 / r_pred, 1.0 / r_true, u_max, r_pred, r_true)])
    assert r_err.item() < 10.0, 'joint matching must pair (75<->80, 13<->10)'
    # no aux: reduces to the plain permutation rmspe
    th_only, _ = matched_tuple_rmse(pred_th, true_th, theta_span=2.1)
    assert torch.isclose(th_only, rmspe(pred_th, true_th, reduce=False), atol=1e-9).all()


def test_wrapped_diff():
    # full-circle semantics: near-opposite angles ARE far apart (no pi-aliasing)
    assert torch.isclose(wrapped_diff(torch.tensor(torch.pi / 2 - 0.01),
                                      torch.tensor(-torch.pi / 2 + 0.01)),
                         torch.tensor(torch.pi - 0.02), atol=1e-9)
    # wrapping across the +-pi seam still takes the short way around
    assert torch.isclose(wrapped_diff(torch.tensor(torch.pi - 0.01),
                                      torch.tensor(-torch.pi + 0.01)),
                         torch.tensor(-0.02), atol=1e-9)


#************************#
#   simulator contract   #
#************************#
def test_sample_contract_shapes():
    sim = NarrowbandSimulator({'segments': 3, 'n_src_range': (2, 2), 'seed': 0})
    b = sim.sample(4)
    M, G, K = 8, sim.grid_size, 3
    assert b['X'].shape[:3] == (4, K, M) and b['X'].dtype == torch.complex128
    assert b['R_hat'].shape == (4, K, M, M)
    assert b['doas'].shape == (4, K, 2) and b['ranges'].shape == (4, K, 2)
    assert b['a_true_grid'].shape == (4, M, G)
    assert b['a_true_doa'].shape == (4, K, M, 2)
    assert b['R_s'].shape == (4, K, 2, 2)
    assert b['positions'].shape == (4, M, 3)
    assert b['imperfect'].shape[0] == 4


def test_snr_convention():
    sim = NarrowbandSimulator({'snr_range': (20.0, 20.0), 'snapshots_range': (5000, 5000),
                               'n_src_range': (1, 1), 'seed': 1})
    b = sim.sample(4)
    p_sig = b['X'].abs().pow(2).mean().item()
    # one source at 20 dB (power 100) through unit-gain array + unit noise: ~101 per sensor
    assert 60.0 < p_sig < 160.0
