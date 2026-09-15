####################################################################################################
#                                          test_archs.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Tests for the differentiable MUSIC layers, the manifold-separation generator, the       #
#          CNM framework and its latent backends, the encoder, and the losses.                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch
import pytest

from cnmmusic.arrays.geometry import ArrayGeometry
from cnmmusic.arrays.steering import steering_matrix, nearfield_steering_matrix, gauge_fix
from cnmmusic.data.simulator import NarrowbandSimulator
from cnmmusic.archs.music import DifferentiableMUSIC, DifferentiableRootMUSIC
from cnmmusic.archs.encoder import CovarianceEncoder, LagCovarianceEncoder
from cnmmusic.archs.correction import SteeringCorrection
from cnmmusic.archs.manifold import SeparatedManifold
from cnmmusic.criteria.losses import (alignment_loss, ranking_loss, cov_recon_loss,
                                      smoothness_loss, doa_loss, subspace_ratio)
from cnmmusic.criteria.metrics import rmspe, matched_tuple_rmse


#*************#
#   helpers   #
#*************#
THETA_RANGE = (-1.05, 1.05)                        # +-60 deg: accuracy tests avoid endfire


def batch(seed=3, **overrides):
    cfg = {'snr_range': (10.0, 10.0), 'snapshots_range': (200, 200), 'seed': seed,
           'theta_range': THETA_RANGE}
    cfg.update(overrides)
    return NarrowbandSimulator(cfg).sample(8)


#*******************************#
#   differentiable grid music   #
#*******************************#
def test_diff_music_recovers_doas():
    b = batch()
    geom = ArrayGeometry.ula(8)
    music = DifferentiableMUSIC(grid_size=721)
    A = steering_matrix(geom.positions[None], music.grid[None])[0]
    doas, P = music(b['R_hat'][:, 0], A[None], int(b['n_src'][0]))
    err = torch.rad2deg(rmspe(doas, b['doas'][:, 0])).item()
    assert err < 1.5


def test_diff_music_gradient_flows_to_manifold():
    b = batch()
    music = DifferentiableMUSIC(grid_size=181)
    geom = ArrayGeometry.ula(8)
    A = steering_matrix(geom.positions[None], music.grid[None])[0][None].clone()
    A.requires_grad_(True)
    doas, P = music(b['R_hat'][:, 0], A, int(b['n_src'][0]))
    doas.sum().backward()
    assert A.grad is not None and A.grad.abs().sum() > 0


#*******************************#
#   differentiable root-music   #
#*******************************#
def test_rootmusic_matches_grid_music():
    b = batch()
    music = DifferentiableMUSIC(grid_size=1441)
    root = DifferentiableRootMUSIC()
    geom = ArrayGeometry.ula(8)
    A = steering_matrix(geom.positions[None], music.grid[None])[0]
    En = music.noise_subspace(b['R_hat'][:, 0], int(b['n_src'][0]))
    doas_g, _ = music(b['R_hat'][:, 0], A[None], int(b['n_src'][0]))
    doas_r = root(En, int(b['n_src'][0]))
    err = torch.rad2deg(rmspe(doas_r, b['doas'][:, 0])).item()
    assert err < 1.0
    # median parity with the grid scan (the soft-argmax window can drag closely spaced grid
    # peaks toward each other, so the max diff is not a fair comparison)
    assert torch.rad2deg((doas_g - doas_r).abs().median()).item() < 0.5


def test_rooting_gradcheck():
    """The load-bearing check: gradients flow stably through companion-matrix rooting."""
    torch.manual_seed(0)
    root = DifferentiableRootMUSIC()

    b = batch(seed=9)
    music = DifferentiableMUSIC()
    En0 = music.noise_subspace(b['R_hat'][0, 0], int(b['n_src'][0]))

    def fn(re, im):
        En = torch.complex(re, im)
        return root(En[None], 2).sum()

    re = En0.real.clone().requires_grad_(True)
    im = En0.imag.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(fn, (re, im), eps=1e-6, atol=1e-4, nondet_tol=1e-5)


#***********************************#
#   manifold-separation generator   #
#***********************************#
def fitted_manifold(z_dim=8, **kw):
    geom = ArrayGeometry.ula(8)
    man = SeparatedManifold(M=8, z_dim=z_dim, **kw)
    man.fit_nominal(geom.positions)
    return geom, man


def test_manifold_zero_init_is_nominal():
    geom, man = fitted_manifold()
    theta = torch.linspace(-1.2, 1.2, 61, dtype=torch.float64)
    A = man(theta, torch.randn(4, 8))
    A0 = gauge_fix(steering_matrix(geom.positions[None], theta[None])[0])
    align = (A0.conj() * A).sum(1).abs() / (A0.norm(dim=0) * A.norm(dim=1))
    assert A.shape == (4, 8, 61)
    assert (align > 0.999).all(), 'zero-init manifold must reproduce nominal physics'


def test_manifold_rooting_recovers_doas():
    """Untrained CNM = classical root-MUSIC on the wavefield basis."""
    b = batch()
    _, man = fitted_manifold()
    root = DifferentiableRootMUSIC(variable='theta', theta_lim=THETA_RANGE[1] + 0.05)
    music = DifferentiableMUSIC()
    z = torch.randn(8, 8)
    En = music.noise_subspace(b['R_hat'][:, 0], int(b['n_src'][0]))
    pred = root(man.theta_matrix(z).mH @ En, int(b['n_src'][0]))
    err = torch.rad2deg(rmspe(pred, b['doas'][:, 0])).item()
    assert err < 2.0


def test_manifold_conditions_on_z():
    _, man = fitted_manifold()
    with torch.no_grad():                                   # break the zero init
        man.head.weight.normal_(0, 0.1)
    z1, z2 = torch.randn(1, 8), torch.randn(1, 8)
    assert not torch.allclose(man.theta_matrix(z1), man.theta_matrix(z2), atol=1e-6), \
        'different z must give different sampling matrices'


def test_manifold_gradient_flows_z_to_roots():
    b = batch()
    _, man = fitted_manifold()
    with torch.no_grad():
        man.head.weight.normal_(0, 0.01)
    root = DifferentiableRootMUSIC(variable='theta', theta_lim=THETA_RANGE[1] + 0.05)
    music = DifferentiableMUSIC()
    z = torch.randn(8, 8, requires_grad=True)
    En = music.noise_subspace(b['R_hat'][:, 0], int(b['n_src'][0]))
    pred = root(man.theta_matrix(z).mH @ En, int(b['n_src'][0]))
    pred.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_manifold_rf_axes():
    """Active r/f axes change the generated vector; per-axis matrices have the right shape."""
    _, man = fitted_manifold(L_r=4, L_f=4, u_max=0.2, f_halfspan=0.2)
    z = torch.randn(2, 8)
    theta = torch.tensor([0.2, 0.2], dtype=torch.float64)
    A = man(theta, z, r=torch.tensor([10.0, 40.0], dtype=torch.float64))
    corr = (A[0, :, 0].conj() @ A[0, :, 1]).abs() / 8
    assert corr < 0.999, 'range must change the generated vector'
    assert man.r_matrix(z, theta[None].expand(2, 2)).shape == (2, 2, 8, 4)
    assert man.f_matrix(z, theta[None].expand(2, 2)).shape == (2, 2, 8, 4)


#*******************#
#   cnm framework   #
#*******************#
def test_cnm_zero_init_is_nominal():
    """The structural fallback: untrained correction == exact nominal physics."""
    from cnmmusic.models.frameworkCNM import CNMFramework
    fw = CNMFramework(M=8, z_dim=8, hidden=32, theta_range=THETA_RANGE)
    b = batch()
    z = fw.encode(b)
    th, r = b['doas'][:, 0], b['ranges'][:, 0]
    f = torch.ones_like(th)
    a = fw.steer(z, b['positions'], th, r, f)
    a0 = nearfield_steering_matrix(b['positions'], th, r, f)
    assert torch.allclose(a, a0), 'zero-init correction must reproduce the nominal manifold'


def test_cnm_framework_trains():
    from cnmmusic.models.frameworkCNM import CNMFramework
    fw = CNMFramework(M=8, z_dim=8, hidden=32, theta_range=THETA_RANGE)
    b = batch()
    loss, logs = fw.compute_losses(b)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad.abs().sum() for p in fw.parameters() if p.grad is not None]
    assert len(grads) > 0 and sum(g > 0 for g in grads) > 0
    assert 'l_ortho' in logs and 'l_rank' in logs


@pytest.mark.parametrize('enc', ['snapshot', 'covariance', 'lagcov'])
def test_cnm_encoder_inputs(enc):
    from cnmmusic.models.frameworkCNM import CNMFramework
    fw = CNMFramework(M=8, z_dim=8, hidden=32, theta_range=THETA_RANGE, encoder=enc)
    b = batch()
    loss, _ = fw.compute_losses(b)
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize('enc', ['snapshot', 'covariance'])
def test_cnm_true_cov_exact_at_zero_init(enc):
    """train_En='true' + no imperfections: the noise subspace of the ensemble covariance is
    exactly orthogonal to the (= nominal at zero init) steering vectors, so the orthogonality
    term is zero BY CONSTRUCTION -- the clean-covariance training contract. With train_cov=
    'true' the covariance encoder consumes R_true; loss and gradients must stay finite."""
    from cnmmusic.models.frameworkCNM import CNMFramework
    fw = CNMFramework(M=8, z_dim=8, hidden=32, theta_range=THETA_RANGE,
                      train_cov='true', train_En='true', encoder=enc)
    fw.train()
    b = batch()
    loss, logs = fw.compute_losses(b)
    assert torch.isfinite(loss)
    loss.backward()
    assert logs['l_ortho'].item() < 1e-9, 'true-covariance orthogonality must be exact at init'


def test_cnm_estimate_matches_classical_music():
    """Zero-init CNM estimation == classical MUSIC on the nominal manifold (the fallback)."""
    from cnmmusic.models.frameworkCNM import CNMFramework
    fw = CNMFramework(M=8, z_dim=8, hidden=32, theta_range=THETA_RANGE)
    b = batch()
    d = int(b['n_src'][0])
    z = fw.encode(b)
    En = fw.noise_subspace(b['R_hat'][:, 0], d)
    doas, r_hat, f_hat = fw.estimate(z, En, b['positions'], d)
    err = torch.rad2deg(rmspe(doas, b['doas'][:, 0])).item()
    assert err < 1.5
    assert r_hat is None and f_hat is None


def test_cnm_validation_backends():
    """estimation_metrics scores the corrected manifold through MUSIC (music/) and MVDR
    (mvdr/) plus the nominal-manifold references (music0/, mvdr0/) on the same batch; at
    zero init corrected == nominal, so each estimator must match its reference exactly."""
    from cnmmusic.models.frameworkCNM import CNMFramework
    fw = CNMFramework(M=8, z_dim=8, hidden=32, theta_range=THETA_RANGE)
    b = batch()
    m = fw.estimation_metrics(b)
    assert m['music/doa_rmspe_deg'].item() < 1.5
    assert torch.isfinite(m['mvdr/doa_rmspe_deg'])
    assert m['mvdr/doa_rmspe_deg'].item() < 15.0   # MVDR resolves worse than MUSIC, sane only
    assert torch.allclose(m['music/doa_rmspe_deg'], m['music0/doa_rmspe_deg'])
    assert torch.allclose(m['mvdr/doa_rmspe_deg'], m['mvdr0/doa_rmspe_deg'])
    assert torch.allclose(m['root/doa_rmspe_deg'], m['root0/doa_rmspe_deg'])
    assert m['root0/doa_rmspe_deg'].item() < 2.0   # classical root-MUSIC on an easy ULA batch
    assert torch.allclose(m['sbl/doa_rmspe_deg'], m['sbl0/doa_rmspe_deg'])
    assert m['sbl0/doa_rmspe_deg'].item() < 3.0    # SBL1 (Gerstoft 2016) on an easy ULA batch


def test_cnm_val_estimators_configurable():
    from cnmmusic.models.frameworkCNM import CNMFramework
    fw = CNMFramework(M=8, z_dim=8, hidden=32, theta_range=THETA_RANGE,
                      val_estimators=('music',))
    m = fw.estimation_metrics(batch())
    assert set(m) == {'music/doa_rmspe_deg', 'music0/doa_rmspe_deg'}


@pytest.mark.parametrize('backend', ['mlp', 'sphere'])
def test_cnm_framework_latent_backends(backend):
    from cnmmusic.models.frameworkCNM import CNMFramework
    fw = CNMFramework(M=8, z_dim=8, hidden=32, theta_range=THETA_RANGE, backend=backend)
    b = batch()
    loss, logs = fw.compute_losses(b)
    assert torch.isfinite(loss)
    loss.backward()
    if backend == 'sphere':
        assert torch.allclose(fw.encode(b).norm(dim=-1),
                              torch.ones(b['X'].shape[0])), 'sphere latent must be unit norm'


def test_cnm_hard_negative_mining():
    """n_hard > 0 mines the deepest spurious nulls as extra negatives; loss stays finite and
    differentiable, and n_hard = 0 leaves the loss path untouched."""
    from cnmmusic.models.frameworkCNM import CNMFramework
    fw = CNMFramework(M=8, z_dim=8, hidden=32, theta_range=THETA_RANGE,
                      n_hard=8, n_probe=64)
    b = batch()
    loss, logs = fw.compute_losses(b)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad.abs().sum() for p in fw.parameters() if p.grad is not None]
    assert sum(g > 0 for g in grads) > 0


def test_cnm_sphere_sampling():
    from cnmmusic.models.frameworkCNM import CNMFramework
    fw = CNMFramework(M=8, z_dim=8, hidden=32, theta_range=THETA_RANGE, backend='sphere')
    b = batch()
    doas = fw.sample_doas(b, n_samples=4, kappa=50.0)
    assert doas.shape[0] == 4 and torch.isfinite(doas).all()
    assert doas.std(dim=0).mean() < 0.5, 'zero-init draws must stay near classical estimates'


def test_damusic_aperture_manifold():
    """The explicit A_grid input: ULA = the paper formula; UCA differs; forward consumes it."""
    from cnmmusic.archs.baselines import DAMUSIC
    net = DAMUSIC(N=8)
    m = torch.arange(8, dtype=torch.float64)
    paper = torch.exp(-1j * torch.pi * m[:, None] * torch.sin(net.grid)[None, :])
    A_ula = steering_matrix(ArrayGeometry.ula(8).positions[None], net.grid[None])[0]
    align = paper.conj().mul(A_ula).sum(0).abs() / 8                    # per-angle alignment
    assert (align > 0.999).all(), 'ULA manifold must match the paper formula (up to phase)'
    A_uca = steering_matrix(ArrayGeometry.uca(8, radius=1.31).positions[None],
                            net.grid[None])[0]
    assert not torch.allclose(A_uca, A_ula, atol=1e-3), \
        'UCA manifold must differ from the ULA formula'
    doas = net(torch.randn(4, 8, 50, dtype=torch.complex128), 2, A_uca)
    assert doas.shape == (4, 7) and torch.isfinite(doas).all()


def test_full_circle_wrapping():
    """Antipodal predictions must NOT count as correct (the pi-wrap bug on planar FOVs)."""
    pred = torch.tensor([[0.0]], dtype=torch.float64)
    true = torch.tensor([[torch.pi]], dtype=torch.float64)
    assert torch.rad2deg(rmspe(pred, true)).item() > 179.0, \
        'a 180-deg error must score as 180 deg, not 0'
    assert doa_loss(pred, true).item() > 9.0                # ~pi^2, not 0


def test_simulator_freeze_geometry():
    """freeze_geometry: one random array drawn per run, identical positions every batch."""
    cfg = {'geometries': ['random_planar'], 'freeze_geometry': True, 'seed': 7,
           'fov_deg': {'random_planar': 180.0}}
    sim = NarrowbandSimulator(cfg)
    p1, p2 = sim.sample(4)['positions'], sim.sample(4)['positions']
    assert torch.equal(p1, p2), 'frozen geometry must be identical across batches'
    sim2 = NarrowbandSimulator({**cfg, 'seed': 8})
    assert not torch.equal(p1, sim2.sample(4)['positions']), \
        'different seeds must give different frozen arrays'


#*********************************#
#   near-field joint estimation   #
#*********************************#
def test_cnm_nearfield_loss_and_estimate():
    """Active range axis: the loss (incl. ghost cross-pairings) trains, and the zero-init joint
    (theta, r) scan matches classical 2-D near-field MUSIC accuracy."""
    from cnmmusic.models.frameworkCNM import CNMFramework
    fw = CNMFramework(M=8, z_dim=8, hidden=32, theta_range=THETA_RANGE,
                      range_span=(5.0, 30.0))
    b = batch(seed=11, range_range=(5.0, 30.0), n_src_range=(2, 2),
              snr_range=(15.0, 15.0), snapshots_range=(300, 300))
    loss, logs = fw.compute_losses(b)
    assert torch.isfinite(loss)
    loss.backward()

    z = fw.encode(b)
    En = fw.noise_subspace(b['R_hat'][:, 0], 2)
    doas, r_hat, _ = fw.estimate(z, En, b['positions'], 2)
    u_hat = 1.0 / r_hat
    u_true = 1.0 / b['ranges'][:, 0]
    th_err, (u_err,) = matched_tuple_rmse(doas, b['doas'][:, 0],
                                          aux=[(u_hat, u_true, fw.u_max)],
                                          theta_span=THETA_RANGE[1] - THETA_RANGE[0])
    assert torch.rad2deg(th_err.pow(2).mean().sqrt()).item() < 3.0
    assert u_err.pow(2).mean().sqrt().item() < 0.05


#*********************************#
#   per-source carriers (tones)   #
#*********************************#
def test_simulator_per_source_freq():
    b = batch(seed=5, per_source_freq=True, freq_range=(0.8, 1.2), n_src_range=(3, 3))
    assert b['freqs'].ndim == 2 and b['freqs'].shape[0] == 8
    f = b['freqs'][:, :3]
    assert torch.isfinite(f).all() and (f >= 0.8).all() and (f <= 1.2).all()
    assert f.std() > 0, 'per-source carriers must differ'


def test_cnm_trains_on_tones_and_reads_freq():
    """Joint (theta, f) on a UCA: (theta, f) is only identifiable for non-linear geometries
    (a far-field ULA tone is exact along the f sin(theta) ridge), so accuracy is scored on
    the UCA; the loss itself must train on any geometry."""
    from cnmmusic.models.frameworkCNM import CNMFramework
    fw = CNMFramework(M=8, z_dim=8, hidden=32, theta_range=THETA_RANGE,
                      freq_span=(0.8, 1.2))
    b = batch(seed=6, geometry='uca', per_source_freq=True, freq_range=(0.8, 1.2),
              n_src_range=(2, 2))
    loss, logs = fw.compute_losses(b)
    assert torch.isfinite(loss) and 'l_ortho' in logs
    loss.backward()

    z = fw.encode(b)
    En = fw.noise_subspace(b['R_hat'][:, 0], 2)
    doas, _, f_hat = fw.estimate(z, En, b['positions'], 2)
    assert f_hat is not None and torch.isfinite(f_hat).all()
    assert (f_hat >= 0.8).all() and (f_hat <= 1.2).all()
    _, (f_err,) = matched_tuple_rmse(doas, b['doas'][:, 0],
                                     aux=[(f_hat, b['freqs'][:, :2], 2 * fw.f_halfspan)],
                                     theta_span=THETA_RANGE[1] - THETA_RANGE[0])
    assert f_err.pow(2).mean().sqrt().item() < 0.1, \
        'zero-init joint (theta, f) scan must land near the true carriers'


#***************************************#
#   basis-domain esprit (conformance)   #
#***************************************#
@pytest.mark.xfail(reason='G+ G is a rank-M projector (L >> M), which destroys the Vandermonde '
                          'shift structure — naive basis-domain ESPRIT does not work; kept as '
                          'the documented negative result behind the experimental cnm-esprit')
def test_manifold_basis_esprit_recovers_doas():
    """Shift invariance in the wavefield domain: ESPRIT via G+ on the signal subspace."""
    b = batch()
    _, man = fitted_manifold()
    z = torch.randn(8, 8)
    d = int(b['n_src'][0])
    _, evecs = torch.linalg.eigh(b['R_hat'][:, 0])
    Es = evecs[..., -d:]
    B = torch.linalg.pinv(man.theta_matrix(z)) @ Es        # (B, L, d) basis-domain subspace
    Phi = torch.linalg.lstsq(B[:, :-1, :], B[:, 1:, :]).solution
    doas = torch.angle(torch.linalg.eigvals(Phi)).sort(dim=-1).values
    err = torch.rad2deg(rmspe(doas, b['doas'][:, 0])).item()
    assert err < 5.0


#*************#
#   encoder   #
#*************#
def test_encoder_shapes_and_scale_invariance():
    b = batch()
    enc = CovarianceEncoder(M=8)
    z = enc(b['R_hat'][:, 0])
    assert z.shape == (8, enc.out_dim)
    feats = enc.features(b['R_hat'][:, 0])
    feats_scaled = enc.features(b['R_hat'][:, 0] * 7.3)
    assert torch.allclose(feats[..., :-1], feats_scaled[..., :-1], atol=1e-9)


def test_lagcov_encoder_any_T():
    """Lags beyond the available snapshots are zeroed: T < tau must still work."""
    enc = LagCovarianceEncoder(M=8, tau=8)
    for T in (1, 3, 50):
        z = enc(torch.randn(4, 8, T, dtype=torch.complex128))
        assert z.shape == (4, enc.out_dim) and torch.isfinite(z).all()


def mixed_T_batch(M=8, T_max=60, lengths=(60, 17, 3, 1)):
    torch.manual_seed(0)
    X = torch.zeros(len(lengths), M, T_max, dtype=torch.complex128)
    for b, T in enumerate(lengths):
        X[b, :, :T] = torch.randn(M, T, dtype=torch.complex128)
    return X, torch.tensor(lengths)


def test_snapshot_encoder_masked_equals_per_element():
    """One packed-GRU call over a zero-padded mixed-T batch must equal running each element
    dense at its own length -- the contract that makes source-count-only grouping exact."""
    from cnmmusic.archs.encoder import SnapshotEncoder
    enc = SnapshotEncoder(M=8, z_dim=16)
    X, lengths = mixed_T_batch()
    z = enc(X, lengths)
    for b, T in enumerate(lengths.tolist()):
        zb = enc(X[b:b + 1, :, :T])
        assert torch.allclose(z[b], zb[0], atol=1e-5), f'mismatch at T={T}'


def test_lagcov_encoder_masked_equals_per_element():
    """Masked lag features over a zero-padded mixed-T batch must equal the dense per-element
    computation (zero padding keeps lag sums exact; each lag divides by its own count)."""
    enc = LagCovarianceEncoder(M=8, tau=8)
    X, lengths = mixed_T_batch()
    z = enc(X, lengths)
    for b, T in enumerate(lengths.tolist()):
        zb = enc(X[b:b + 1, :, :T])
        assert torch.allclose(z[b], zb[0], atol=1e-5), f'mismatch at T={T}'


def test_correction_conditions_on_z():
    corr = SteeringCorrection(M=8, z_dim=8, hidden=32)
    with torch.no_grad():                                   # break the zero init
        corr.net[-1].weight.normal_(0, 0.1)
    th = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)[None]
    u = torch.zeros_like(th)
    f = torch.ones_like(th)
    d1 = corr(torch.randn(1, 8), th, u, f)
    d2 = corr(torch.randn(1, 8), th, u, f)
    assert not torch.allclose(d1, d2, atol=1e-6), \
        'different z must give different corrections'


#************#
#   losses   #
#************#
def test_alignment_loss_gauge_invariant():
    a = torch.randn(4, 8, 21, dtype=torch.complex128)
    scale = torch.polar(torch.rand(4, 1, 21) + 0.5, torch.rand(4, 1, 21) * 6).to(a.dtype)
    assert alignment_loss(a * scale, a).item() < 1e-10
    b_perp = torch.randn_like(a)
    assert alignment_loss(b_perp, a).item() > 0.1


def test_cov_recon_zero_at_truth():
    b = batch()
    loss = cov_recon_loss(b['R_hat'][:, 0],
                          b['a_true_doa'][:, 0] * 0 + b['a_true_doa'][:, 0],
                          b['R_s'][:, 0], b['sigma2'])
    # finite snapshots -> small but nonzero; must be far below a mismatched manifold's loss
    wrong = cov_recon_loss(b['R_hat'][:, 0], torch.randn_like(b['a_true_doa'][:, 0]),
                           b['R_s'][:, 0], b['sigma2'])
    assert loss.item() < 0.2 and loss.item() < wrong.item()


def test_ranking_and_smoothness_finite():
    q_t, q_n = torch.rand(8, 2) * 1e-3, torch.rand(8, 32)
    assert torch.isfinite(ranking_loss(q_t, q_n))
    a = torch.randn(8, 8, 61, dtype=torch.complex128)
    a0 = torch.randn(8, 8, 61, dtype=torch.complex128)
    assert torch.isfinite(smoothness_loss(a, a0))
    assert smoothness_loss(a0, a0).item() < 1e-20        # zero at nominal


def test_nearfield_steering_per_tuple_freq():
    """Per-tuple f (matching theta's last dim) must equal per-tuple scalar-f calls."""
    pos = ArrayGeometry.ula(8).positions[None]
    th = torch.tensor([[0.3, -0.5]], dtype=torch.float64)
    r = torch.tensor([[20.0, torch.inf]], dtype=torch.float64)
    f = torch.tensor([[0.9, 1.1]], dtype=torch.float64)
    A = nearfield_steering_matrix(pos, th, r, f)
    for i in range(2):
        Ai = nearfield_steering_matrix(pos, th[:, i:i + 1], r[:, i:i + 1], float(f[0, i]))
        assert torch.allclose(A[..., i:i + 1], Ai)


def test_subspace_ratio_bounds_and_gauge_invariance():
    torch.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.complex128))
    En = Q[:, :6][None]                                     # orthonormal noise subspace, d = 2
    a_sig = Q[:, 6:8][None]                                 # exactly in the signal span
    a_rand = torch.randn(1, 8, 5, dtype=torch.complex128)
    assert subspace_ratio(En, a_sig).max().item() < 1e-24
    J = subspace_ratio(En, a_rand)
    assert (J >= 0).all() and (J <= 1 + 1e-12).all()
    scale = torch.polar(torch.rand(1, 1, 5) + 0.5, torch.rand(1, 1, 5) * 6).to(a_rand.dtype)
    assert torch.allclose(J, subspace_ratio(En, a_rand * scale), atol=1e-12)


def test_doa_loss_matches_rmspe_and_differentiable():
    pred = torch.tensor([[0.1, -0.3]], dtype=torch.float64, requires_grad=True)
    true = torch.tensor([[-0.31, 0.12]], dtype=torch.float64)
    loss = doa_loss(pred, true)
    loss.backward()
    assert pred.grad is not None
    assert torch.isclose(loss.sqrt(), rmspe(pred.detach(), true), atol=1e-9)
