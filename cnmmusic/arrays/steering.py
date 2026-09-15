####################################################################################################
#                                           steering.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Differentiable steering-vector construction for arbitrary array geometries. Fixes the   #
#          project-wide conventions: sensor positions in half-wavelength units at the carrier      #
#          (standard ULA = [0, 1, ...]), frequencies normalized to the carrier (f = 1), azimuth    #
#          from broadside with u(theta) = [sin, cos, 0], far field a_m = e^(-j pi f p_m^T u), and  #
#          near field a_m = (r / d_m) e^(-j pi f (r - d_m)) with d_m = ||r u - p_m|| (phase at     #
#          the array origin, recovering the far field exactly as r -> inf). Gauge fixing scales    #
#          to ||a|| = sqrt(M) with a real-positive first sensor entry.                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch


#***********************#
#   direction vectors   #
#***********************#
def direction_vector(theta):
    """
    Unit propagation-plane direction(s) for azimuth(s) theta (radians, from broadside).

    theta: (...,) real tensor -> returns (..., 3) with rows [sin(theta), cos(theta), 0].
    """
    sin, cos = torch.sin(theta), torch.cos(theta)
    return torch.stack([sin, cos, torch.zeros_like(sin)], dim=-1)


#************************#
#   far-field steering   #
#************************#
def steering_matrix(positions, theta, f=1.0):
    """
    Far-field steering matrix for arbitrary 3-D geometries.

    positions: (..., M, 3) sensor positions in half-wavelength units
    theta:     (..., G)    azimuths in radians (broadcast-compatible with positions batch dims)
    f:         scalar or (...,) normalized frequency (1 = carrier)

    Returns complex tensor of shape (..., M, G).
    """
    theta = torch.as_tensor(theta, dtype=positions.dtype, device=positions.device)
    u = direction_vector(theta)                                        # (..., G, 3)
    proj = torch.einsum('...mk,...gk->...mg', positions, u)            # (..., M, G)
    f = torch.as_tensor(f, dtype=positions.dtype, device=positions.device)
    if f.ndim > 0:                       # per-angle f (matches theta's last dim) or batch scalar
        f = f[..., None, :] if f.shape[-1] == theta.shape[-1] else f[..., None, None]
    phase = -torch.pi * f * proj
    return torch.polar(torch.ones_like(phase), phase)


#*************************#
#   near-field steering   #
#*************************#
def nearfield_steering_matrix(positions, theta, r, f=1.0):
    """
    Exact spherical-wavefront steering matrix (near-field), phase referenced to the array origin.

    positions: (..., M, 3) sensor positions in half-wavelength units
    theta:     (..., G)    azimuths in radians
    r:         (..., G)    ranges in half-wavelength units (may be inf for far-field entries)
    f:         scalar or (...,) normalized frequency

    Returns complex tensor of shape (..., M, G). Entries with r = inf reduce to far field.
    """
    theta = torch.as_tensor(theta, dtype=positions.dtype, device=positions.device)
    r = torch.as_tensor(r, dtype=positions.dtype, device=positions.device)
    r = r.expand(theta.shape) if r.ndim < theta.ndim else r
    u = direction_vector(theta)                                        # (..., G, 3)

    finite = torch.isfinite(r)
    r_safe = torch.where(finite, r, torch.ones_like(r))

    src = r_safe[..., None] * u                                        # (..., G, 3)
    diff = src[..., None, :, :] - positions[..., :, None, :]           # (..., M, G, 3)
    d = torch.linalg.vector_norm(diff, dim=-1)                        # (..., M, G)

    # spherical path-length difference (r - d_m matches the exp(-j*pi*m*sin) ULA sign convention)
    delay_nf = r_safe[..., None, :] - d
    amp_nf = r_safe[..., None, :] / d.clamp_min(1e-12)

    # far-field limit for entries with r = inf
    proj = torch.einsum('...mk,...gk->...mg', positions, u)
    delay = torch.where(finite[..., None, :], delay_nf, proj)
    amp = torch.where(finite[..., None, :], amp_nf, torch.ones_like(amp_nf))

    f = torch.as_tensor(f, dtype=positions.dtype, device=positions.device)
    if f.ndim > 0:                       # per-angle f (matches theta's last dim) or batch scalar
        f = f[..., None, :] if f.shape[-1] == theta.shape[-1] else f[..., None, None]
    phase = -torch.pi * f * delay
    return torch.polar(amp, phase)


#**************************#
#   steering derivatives   #
#**************************#
def steering_derivative(positions, theta, f=1.0):
    """
    Analytic d a / d theta for the far-field steering matrix (same conventions as
    steering_matrix): da_m = a_m * (-j * pi * f * p_m^T u'(theta)), u' = [cos, -sin, 0].
    positions (..., M, 3), theta (..., G) -> (..., M, G) complex.
    """
    theta = torch.as_tensor(theta, dtype=positions.dtype, device=positions.device)
    a = steering_matrix(positions, theta, f)
    du = torch.stack([torch.cos(theta), -torch.sin(theta), torch.zeros_like(theta)], dim=-1)
    proj = torch.einsum('...mk,...gk->...mg', positions, du)
    f = torch.as_tensor(f, dtype=positions.dtype, device=positions.device)
    if f.ndim > 0:
        f = f[..., None, None]
    return a * (-1j * torch.pi * f * proj)


#******************#
#   gauge fixing   #
#******************#
def gauge_fix(a, dim=-2):
    """
    Remove the arbitrary complex scaling of steering vectors: normalize each vector to
    || a || = sqrt(M) along `dim` and rotate so the first sensor entry is real-positive.
    """
    m = a.shape[dim]
    norm = torch.linalg.vector_norm(a, dim=dim, keepdim=True).clamp_min(1e-12)
    a = a * (m ** 0.5) / norm
    first = a.narrow(dim, 0, 1)
    rot = torch.polar(torch.ones_like(first.real), -torch.angle(first))
    return a * rot
