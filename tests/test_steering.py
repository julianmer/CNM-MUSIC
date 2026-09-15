####################################################################################################
#                                         test_steering.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Unit tests for steering vectors, geometries, and imperfection models.                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch
import pytest

from cnmmusic.arrays.geometry import ArrayGeometry, GeometrySampler
from cnmmusic.arrays.steering import (steering_matrix, nearfield_steering_matrix, gauge_fix,
                                      direction_vector)
from cnmmusic.arrays.imperfections import ImperfectionModel, ImperfectionParams


#************************#
#   far-field steering   #
#************************#
def test_ula_matches_analytic():
    geom = ArrayGeometry.ula(8)
    theta = torch.linspace(-1.2, 1.2, 21, dtype=torch.float64)
    a = steering_matrix(geom.positions[None], theta[None])[0]
    m = torch.arange(8, dtype=torch.float64)[:, None]
    ref = torch.exp(-1j * torch.pi * m * torch.sin(theta)[None, :])
    assert torch.allclose(a, ref, atol=1e-12)


def test_frequency_scaling():
    geom = ArrayGeometry.ula(6)
    theta = torch.tensor([0.4], dtype=torch.float64)
    a_half = steering_matrix(geom.positions[None], theta[None], f=0.5)[0]
    a_ref = steering_matrix((0.5 * geom.positions)[None], theta[None], f=1.0)[0]
    assert torch.allclose(a_half, a_ref, atol=1e-12)


def test_batched_shapes():
    pos = torch.randn(4, 7, 3, dtype=torch.float64)
    theta = torch.randn(4, 11, dtype=torch.float64)
    a = steering_matrix(pos, theta)
    assert a.shape == (4, 7, 11) and a.dtype == torch.complex128


#*************************#
#   near-field steering   #
#*************************#
def test_nearfield_farfield_limit():
    geom = ArrayGeometry.ula(8)
    theta = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)
    far = steering_matrix(geom.positions[None], theta[None])[0]
    near = nearfield_steering_matrix(geom.positions[None], theta[None],
                                     torch.full_like(theta, 1e7)[None])[0]
    assert torch.allclose(near, far, atol=1e-4)


def test_nearfield_inf_exact_farfield():
    geom = ArrayGeometry.ula(5)
    theta = torch.tensor([0.3, -0.7], dtype=torch.float64)
    far = steering_matrix(geom.positions[None], theta[None])[0]
    near = nearfield_steering_matrix(geom.positions[None], theta[None],
                                     torch.full_like(theta, torch.inf)[None])[0]
    assert torch.allclose(near, far, atol=1e-12)


def test_nearfield_range_distinguishes():
    geom = ArrayGeometry.ula(8)
    theta = torch.tensor([0.2, 0.2], dtype=torch.float64)
    r = torch.tensor([10.0, 40.0], dtype=torch.float64)
    a = nearfield_steering_matrix(geom.positions[None], theta[None], r[None])[0]
    corr = (a[:, 0].conj() @ a[:, 1]).abs() / (a[:, 0].norm() * a[:, 1].norm())
    assert corr < 0.999                       # same bearing, different range -> different vectors


#******************#
#   gauge fixing   #
#******************#
def test_gauge_fix_properties():
    a = torch.randn(6, 13, dtype=torch.complex128)
    g = gauge_fix(a)
    assert torch.allclose(torch.linalg.vector_norm(g, dim=-2),
                          torch.full((13,), 6 ** 0.5, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(g[0].imag, torch.zeros(13, dtype=torch.float64), atol=1e-10)
    assert (g[0].real > 0).all()
    assert torch.allclose(gauge_fix(g), g, atol=1e-10)               # idempotent
    # invariant to arbitrary complex scaling
    scale = torch.polar(torch.rand(13, dtype=torch.float64) + 0.5,
                        torch.rand(13, dtype=torch.float64) * 6)
    assert torch.allclose(gauge_fix(a * scale), g, atol=1e-9)


#*******************#
#   imperfections   #
#*******************#
def test_rho_zero_is_identity():
    geom = ArrayGeometry.ula(8)
    model = ImperfectionModel(rho=0.0, randomized=False)
    params = model.sample(8)
    theta = torch.linspace(-1.0, 1.0, 7, dtype=torch.float64)
    a_true = model.true_manifold(geom, params, theta)
    a0 = gauge_fix(steering_matrix(geom.positions[None], theta[None])[0])
    assert torch.allclose(a_true, a0, atol=1e-10)


def test_coupling_toeplitz():
    model = ImperfectionModel(kinds=('coupling',), rho=1.0, randomized=False)
    params = model.sample(6)
    C = params.coupling
    for k in range(1, 6):
        vals = torch.diagonal(C, offset=k)
        assert torch.allclose(vals, vals[0].expand_as(vals), atol=1e-12)
        assert torch.allclose(vals[0], model.gamma ** k, atol=1e-12)
    assert torch.allclose(C, C.T, atol=1e-12)                        # symmetric Toeplitz


def test_imperfection_changes_manifold():
    geom = ArrayGeometry.ula(8)
    model = ImperfectionModel(rho=1.0, randomized=True, seed=3)
    params = model.sample(8)
    theta = torch.linspace(-1.0, 1.0, 7, dtype=torch.float64)
    a_true = model.true_manifold(geom, params, theta)
    a0 = gauge_fix(steering_matrix(geom.positions[None], theta[None])[0])
    align = (a_true.conj() * a0).sum(0).abs() / 8
    assert (align < 0.999).all()                                     # visibly perturbed


def test_gauge_reference_sensor():
    model = ImperfectionModel(rho=1.0, randomized=True, seed=1)
    p = model.sample(8)
    assert p.delta_pos[0].abs().sum() == 0 and p.phase[0] == 0 and p.gain[0] == 1


#**************#
#   geometry   #
#**************#
def test_geometry_factories():
    assert ArrayGeometry.ula(8).M == 8
    assert abs(ArrayGeometry.ula(8).aperture - 7.0) < 1e-9
    assert ArrayGeometry.uca(8, radius=2.0).M == 8
    assert ArrayGeometry.ura(4, 2).M == 8
    g = ArrayGeometry.random_planar(8, aperture=7.0, rng=torch.Generator().manual_seed(0))
    assert g.M == 8 and abs(g.aperture - 7.0) < 1e-6


def test_geometry_sampler():
    s = GeometrySampler(kinds=('ula', 'uca', 'nula', 'random_planar'), M_range=(6, 10),
                        aperture_range=(5.0, 9.0), seed=0)
    for _ in range(10):
        g = s.sample()
        assert 6 <= g.M <= 10
