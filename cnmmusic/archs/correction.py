####################################################################################################
#                                          correction.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 20/07/26                                                                                #
#                                                                                                  #
# Purpose: The observation-conditioned steering-vector correction:                                 #
#          a_hat(theta, r, f | z) = a0(theta, r, f) * (1 + Delta(z, theta, u, f)) with the exact   #
#          nominal (near-field, per-tuple-frequency) manifold a0 built by the caller and a         #
#          ZERO-INIT complex head on Delta -- at init and for untrained checkpoints the model IS   #
#          classical physics, so every downstream estimator falls back to its textbook self.       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch
import torch.nn as nn


#**************************************************************************************************#
#                                     Class SteeringCorrection                                     #
#**************************************************************************************************#
#                                                                                                  #
# A small conditional neural field over the tuple (theta, u, f). Every axis is always present      #
# and identically structured: standardized onto [-1, 1] over its configured span (a pinned axis    #
# has span None and contributes constants), with per-axis comb frequencies matched to the span     #
# (theta in raw radians, extras on the standardized coordinate). Two orthogonal switches:          #
#                                                                                                  #
# coord_enc -- how the tuple enters ('harm' fixed sin/cos comb | 'lff' learnable Fourier           #
# features, comb-initialized | 'raw' standardized scalars only).                                   #
#                                                                                                  #
# cond_mode -- how the scene conditions the field ('concat' z into the first layer | 'film'        #
# z -> per-hidden-layer scale/shift with zero-init modulations | 'attn' per-tuple cross-           #
# attention over encoder tokens plus a global z token, zero-init output projection).               #
#                                                                                                  #
# The last layer is zero-initialized unless zero_init=False (corr_mode 'free'): Delta == 0         #
# until training moves it.                                                                         #
#                                                                                                  #
#**************************************************************************************************#
class SteeringCorrection(nn.Module):
    def __init__(self, M=8, z_dim=16, hidden=128, n_harm=4, zero_init=True,
                 cond_mode='concat', coord_enc='harm',
                 theta_range=(-1.047, 1.047), u_span=None, f_span=None,
                 n_tok=8, d_tok=16):
        super().__init__()
        assert cond_mode in ('concat', 'film', 'attn'), f'unknown cond_mode: {cond_mode}'
        assert coord_enc in ('harm', 'lff', 'raw'), f'unknown coord_enc: {coord_enc}'
        self.M, self.n_harm, self.z_dim = M, n_harm, z_dim
        self.cond_mode, self.coord_enc = cond_mode, coord_enc
        self.theta_range = tuple(theta_range)
        self.u_span, self.f_span = u_span, f_span
        self.n_tok, self.d_tok = n_tok, d_tok

        if coord_enc == 'harm':                  # the same pi-comb on every standardized axis
            coord_dim = 3 * (2 * n_harm + 1)
        elif coord_enc == 'lff':                 # one learnable bank over all three axes,
            n_freq = 3 * n_harm                  # initialized exactly at the harm comb
            self.lff = nn.Linear(3, n_freq)
            with torch.no_grad():
                self.lff.weight.zero_()
                self.lff.bias.zero_()
                hi = max(abs(theta_range[0]), abs(theta_range[1]))
                for k in range(n_harm):
                    self.lff.weight[k, 0] = (k + 1) * hi
                for a in (1, 2):
                    for k in range(n_harm):
                        self.lff.weight[a * n_harm + k, a] = (k + 1) * torch.pi
            coord_dim = 2 * n_freq + 3
        else:
            coord_dim = 3

        in_dim = coord_dim + (z_dim if cond_mode == 'concat' else 0)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2 * M))
        if cond_mode == 'film':                  # z -> (gamma, beta) per hidden layer; the
            self.film = nn.Sequential(nn.Linear(z_dim, 64), nn.SiLU(),   # zero-init head makes
                                      nn.Linear(64, 4 * hidden))         # every modulation start
            nn.init.zeros_(self.film[-1].weight)                         # as identity (AdaLN-Zero)
            nn.init.zeros_(self.film[-1].bias)
        if cond_mode == 'attn':                  # per-sensor tokens (+ global z token); the
            self.tok_in = nn.Linear(d_tok + 3, hidden)     # zero-init out_proj makes the block
            self.z_tok = nn.Linear(z_dim, hidden)          # start as identity on the trunk
            self.attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
            nn.init.zeros_(self.attn.out_proj.weight)
            nn.init.zeros_(self.attn.out_proj.bias)
            self.ln_q = nn.LayerNorm(hidden)
            self.ln_t = nn.LayerNorm(hidden)
        if zero_init:                            # Delta == 0 at init (mult/add anchor on a0);
            nn.init.zeros_(self.net[-1].weight)  # 'free' mode keeps the standard init --
            nn.init.zeros_(self.net[-1].bias)    # a zero manifold would make J = 0/eps

    @staticmethod
    def _std(x, span):
        """Affine map of x from (lo, hi) onto exactly [-1, 1]; constant 0 if span is None."""
        if span is None:
            return torch.zeros_like(x)
        lo, hi = span
        return 2.0 * (x - lo) / max(hi - lo, 1e-12) - 1.0

    def features(self, theta, u, f):
        """theta/u/f (..., n) -> real coordinate features (..., n, coord_dim), float32.
        Every axis identically: standardize onto [-1, 1] (pinned span -> constant 0), then
        the coord_enc encoding -- no per-axis special cases."""
        x = torch.stack([self._std(theta, self.theta_range),
                         self._std(u, self.u_span),
                         self._std(f, self.f_span)], dim=-1)              # (..., n, 3)
        if self.coord_enc == 'raw':
            return x.to(torch.float32)
        if self.coord_enc == 'harm':
            k = torch.arange(1, self.n_harm + 1, device=x.device, dtype=x.dtype)
            ang = torch.cat([theta[..., None] * k,                        # physics comb (radians)
                             x[..., 1:2] * (torch.pi * k),                # pi-comb on the
                             x[..., 2:3] * (torch.pi * k)], dim=-1)       # standardized extras
            return torch.cat([torch.sin(ang), torch.cos(ang), x], dim=-1).to(torch.float32)
        w = self.lff(x.to(self.lff.weight.dtype))
        return torch.cat([torch.sin(w), torch.cos(w), x.to(w.dtype)],
                         dim=-1).to(torch.float32)

    def forward(self, z, theta, u, f, positions=None):
        """z (B, z_dim [+ n_tok * d_tok for attn]), theta/u/f (B, n), positions (B, M, 3)
        -> complex correction Delta (B, M, n)."""
        tokens = None
        if self.cond_mode == 'attn':
            z, flat = z[..., :self.z_dim], z[..., self.z_dim:]
            tokens = flat.reshape(*flat.shape[:-1], self.n_tok, self.d_tok)
        feat = self.features(theta, u, f)                                  # (B, n, F)
        if self.cond_mode == 'concat':
            h = torch.cat([z[:, None, :].expand(-1, feat.shape[1], -1).to(feat.dtype), feat],
                          dim=-1)
            out = self.net(h)
        elif self.cond_mode == 'film':
            g1, b1, g2, b2 = self.film(z)[:, None, :].chunk(4, dim=-1)
            h = self.net[1]((1.0 + g1) * self.net[0](feat) + b1)
            h = self.net[3]((1.0 + g2) * self.net[2](h) + b2)
            out = self.net[4](h)
        else:
            pos = positions[..., :3].to(feat.dtype)
            pos = pos / pos.norm(dim=-1, keepdim=True).amax(dim=-2, keepdim=True).clamp_min(1e-9)
            tok = self.tok_in(torch.cat([tokens.to(feat.dtype), pos], dim=-1))
            tok = torch.cat([tok, self.z_tok(z.to(feat.dtype))[:, None, :]], dim=-2)
            tok = self.ln_t(tok)                                           # (B, n_tok + 1, H)
            h = self.net[1](self.net[0](feat))
            att, _ = self.attn(self.ln_q(h), tok, tok, need_weights=False)
            h = h + att                                                    # identity at init
            out = self.net[4](self.net[3](self.net[2](h)))
        return torch.complex(out[..., :self.M], out[..., self.M:]).transpose(-2, -1)
