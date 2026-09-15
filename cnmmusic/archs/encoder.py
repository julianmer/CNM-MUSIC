####################################################################################################
#                                            encoder.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Observation encoders h(X): the input REPRESENTATION only (GRU over raw snapshots,       #
#          covariance features, lag-covariance features), each exposing out_dim; all latent        #
#          mixing lives in the framework's backend and the correction net.                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch
import torch.nn as nn


#**************************************************************************************************#
#                                     Class CovarianceEncoder                                      #
#**************************************************************************************************#
#                                                                                                  #
# Feature extractor, no learned layers: the upper triangle (real/imag) of the trace-normalized     #
# covariance plus its log-trace, fed directly to the latent backend / correction net (which do     #
# all the mixing). out_dim = 2 * M(M+1)/2 + 1.                                                     #
#                                                                                                  #
#**************************************************************************************************#
class CovarianceEncoder(nn.Module):
    def __init__(self, M=8, **kwargs):
        super().__init__()
        self.M = M
        idx = torch.triu_indices(M, M)
        self.register_buffer('iu', idx[0])
        self.register_buffer('ju', idx[1])
        self.out_dim = 2 * idx.shape[1] + 1                        # Re/Im upper triangle + log-trace

    def features(self, R):
        """R (..., M, M) complex -> real feature vector (..., 2 * M(M+1)/2 + 1)."""
        tr = R.diagonal(dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-12)
        Rn = R / tr[..., None, None]
        upper = Rn[..., self.iu, self.ju]
        return torch.cat([upper.real, upper.imag, torch.log(tr)[..., None]], dim=-1)

    def forward(self, R):
        return self.features(R).to(torch.float32)


#**************************************************************************************************#
#                                      Class SnapshotEncoder                                       #
#**************************************************************************************************#
#                                                                                                  #
# GRU over raw snapshots (no covariance bottleneck): handles ANY number of snapshots (1..1000+),   #
# preserves temporal structure (coherence, nonstationarity) that second-order statistics discard.  #
# Inputs are RMS-normalized with the log-power appended so the 60 dB SNR span stays in range.      #
#                                                                                                  #
#**************************************************************************************************#
class SnapshotEncoder(nn.Module):
    def __init__(self, M=8, z_dim=16, hidden=128):
        super().__init__()
        self.gru = nn.GRU(2 * M, hidden, batch_first=True)
        self.head = nn.Linear(hidden + 1, z_dim)
        self.out_dim = z_dim

    def forward(self, X, lengths=None):
        """
        Complex snapshots (B, M, T), zero-padded past each element's own T_b given by
        lengths (B,) -> z (B, z_dim). Packed sequences make one GRU call over mixed T exact:
        the hidden state is read at each element's last VALID step, and the RMS/log-power
        statistics count only valid samples (zero padding contributes nothing to the sums).
        """
        B, M, T = X.shape
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=X.device)
        lengths = lengths.reshape(-1).clamp(1, T)
        power = X.abs().pow(2).sum(dim=(-2, -1)) / (lengths * M).to(X.real.dtype)
        rms = power.sqrt().clamp_min(1e-12)                                # (B,)
        Xn = X / rms[:, None, None]
        h = torch.cat([Xn.real, Xn.imag], dim=1).transpose(1, 2).to(torch.float32)  # (B, T, 2M)
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, hn = self.gru(packed)
        logp = torch.log(rms).to(torch.float32)[:, None]
        return self.head(torch.cat([hn[0], logp], dim=-1))


#**************************************************************************************************#
#                                   Class LagCovarianceEncoder                                     #
#**************************************************************************************************#
#                                                                                                  #
# Feature extractor, no learned layers: SubspaceNet-style input features (Shmuel, Merkofer et      #
# al., TVT 2024) -- tau time-lagged autocorrelation matrices, RMS-normalized with the log-power    #
# appended (60 dB SNR span), fed directly to the latent backend / correction net. Lags beyond      #
# the available snapshots are zero (T >= 1 always works). out_dim = tau * 2M * M + 1.              #
#                                                                                                  #
#**************************************************************************************************#
class LagCovarianceEncoder(nn.Module):
    def __init__(self, M=8, tau=8, **kwargs):
        super().__init__()
        self.M, self.tau = M, tau
        self.out_dim = tau * 2 * M * M + 1

    def features(self, X, lengths=None):
        """
        Complex snapshots (B, M, T), zero-padded past each element's own T_b given by
        lengths (B,) -> (B, tau * 2M * M + 1) real features. Zero padding keeps the lag sums
        exact; each lag divides by its element's own valid count T_b - i (zero for i >= T_b).
        """
        B, M, T = X.shape
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=X.device)
        lengths = lengths.reshape(-1).clamp(1, T)
        power = X.abs().pow(2).sum(dim=(-2, -1)) / (lengths * M).to(X.real.dtype)
        rms = power.sqrt().clamp_min(1e-12)
        Xn = X / rms[:, None, None]
        feats = []
        for i in range(self.tau):
            if i < T:
                n = (lengths - i).clamp_min(1).to(X.real.dtype)[:, None, None]
                Ri = Xn[..., :T - i] @ Xn[..., i:].mH / n
                Ri = Ri * (lengths > i)[:, None, None].to(Ri.dtype)
            else:
                Ri = torch.zeros(B, self.M, self.M, dtype=X.dtype, device=X.device)
            feats.append(torch.cat([Ri.real, Ri.imag], dim=-2))
        h = torch.stack(feats, dim=1).flatten(1).to(torch.float32)
        logp = torch.log(rms).to(torch.float32)[:, None]
        return torch.cat([h, logp], dim=-1)

    def forward(self, X, lengths=None):
        return self.features(X, lengths)


#**************************************************************************************************#
#                                     Class SnapAttnEncoder                                        #
#**************************************************************************************************#
#                                                                                                  #
# Attention tokenizer over raw snapshots -- no recurrence, no covariance bottleneck, no scenario   #
# gating: the sinusoidal time positional encoding is always on, so temporal structure is visible   #
# wherever the data has it and ignorable where it does not. Handles any T (1..1000+) via key       #
# padding. Each snapshot token carries the raw Re/Im plus its rank-1 outer-product features (so    #
# mean pooling can represent R_hat exactly); K learned seed queries cross-attend over the tokens.  #
# Output is the K scene tokens flattened with the log-power appended: out_dim = K * d + 1. The     #
# attn corr_cond taps them as condition tokens; concat / film consume the pooled latent exactly    #
# as with every other encoder.                                                                     #
#                                                                                                  #
#**************************************************************************************************#
class SnapAttnEncoder(nn.Module):
    def __init__(self, M=8, n_tok=8, d_tok=64, n_heads=4, time_pe=True, n_pe=8, **kwargs):
        super().__init__()
        self.M, self.n_tok, self.d_tok, self.time_pe = M, n_tok, d_tok, time_pe
        idx = torch.triu_indices(M, M)
        self.register_buffer('iu', idx[0])
        self.register_buffer('ju', idx[1])
        in_dim = 2 * M + 2 * idx.shape[1]              # Re/Im snapshot + Re/Im rank-1 triu
        if time_pe:                                    # sinusoidal snapshot-index encoding:
            self.register_buffer('omega',              # order carries e^{j pi f n} for tones,
                                 torch.pi / 2.0 ** torch.arange(n_pe, dtype=torch.float32))
            in_dim += 2 * n_pe                         # geometric ladder pi .. pi/2^(n_pe-1)
        self.embed = nn.Linear(in_dim, d_tok)
        self.seeds = nn.Parameter(torch.randn(n_tok, d_tok) / d_tok ** 0.5)
        self.pool = nn.MultiheadAttention(d_tok, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_tok)
        self.out_dim = n_tok * d_tok + 1

    def forward(self, X, lengths=None):
        """Complex snapshots (B, M, T), zero-padded past lengths (B,) -> (B, n_tok * d_tok + 1)."""
        B, M, T = X.shape
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=X.device)
        lengths = lengths.reshape(-1).clamp(1, T)
        power = X.abs().pow(2).sum(dim=(-2, -1)) / (lengths * M).to(X.real.dtype)
        rms = power.sqrt().clamp_min(1e-12)
        Xn = (X / rms[:, None, None]).transpose(1, 2)                     # (B, T, M)
        outer = Xn[..., :, None] * Xn[..., None, :].conj()                # (B, T, M, M)
        up = outer[..., self.iu, self.ju]                                 # (B, T, P)
        feats = [Xn.real, Xn.imag, up.real, up.imag]
        if self.time_pe:
            ang = torch.arange(T, device=X.device, dtype=torch.float32)[:, None] * self.omega
            feats.append(torch.cat([torch.sin(ang), torch.cos(ang)],
                                   dim=-1)[None].expand(B, -1, -1))
        feats = torch.cat([f.to(torch.float32) for f in feats], dim=-1)
        tok = self.embed(feats)                                           # (B, T, d)
        pad = torch.arange(T, device=X.device)[None, :] >= lengths[:, None]
        seeds = self.seeds[None].expand(B, -1, -1)
        pooled, _ = self.pool(seeds, tok, tok, key_padding_mask=pad, need_weights=False)
        pooled = self.norm(pooled + seeds)                                # (B, n_tok, d)
        logp = torch.log(rms).to(torch.float32)[:, None]
        return torch.cat([pooled.flatten(1), logp], dim=-1)


#**************************************************************************************************#
#                                      Class MeanSetEncoder                                        #
#**************************************************************************************************#
#                                                                                                  #
# Minimal any-T encoder (deep sets): encode each snapshot with a small MLP, masked-mean over the   #
# valid snapshots, append the log-power. Each snapshot token carries the raw Re/Im, its rank-1     #
# outer-product features (so the mean can represent R_hat exactly), and the sinusoidal time        #
# encoding (so the mean can represent DFT / lag statistics). The framework's latent backend does   #
# all further mixing. out_dim = d_phi + 1.                                                         #
#                                                                                                  #
#**************************************************************************************************#
class MeanSetEncoder(nn.Module):
    def __init__(self, M=8, d_phi=128, n_pe=8, **kwargs):
        super().__init__()
        self.M = M
        idx = torch.triu_indices(M, M)
        self.register_buffer('iu', idx[0])
        self.register_buffer('ju', idx[1])
        self.register_buffer('omega',
                             torch.pi / 2.0 ** torch.arange(n_pe, dtype=torch.float32))
        in_dim = 2 * M + 2 * idx.shape[1] + 2 * n_pe
        self.phi = nn.Sequential(nn.Linear(in_dim, d_phi), nn.SiLU(),
                                 nn.Linear(d_phi, d_phi))
        self.out_dim = d_phi + 1

    def forward(self, X, lengths=None):
        """Complex snapshots (B, M, T), zero-padded past lengths (B,) -> (B, d_phi + 1)."""
        B, M, T = X.shape
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=X.device)
        lengths = lengths.reshape(-1).clamp(1, T)
        power = X.abs().pow(2).sum(dim=(-2, -1)) / (lengths * M).to(X.real.dtype)
        rms = power.sqrt().clamp_min(1e-12)
        Xn = (X / rms[:, None, None]).transpose(1, 2)                     # (B, T, M)
        outer = Xn[..., :, None] * Xn[..., None, :].conj()                # (B, T, M, M)
        up = outer[..., self.iu, self.ju]                                 # (B, T, P)
        ang = torch.arange(T, device=X.device, dtype=torch.float32)[:, None] * self.omega
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)[None].expand(B, -1, -1)
        feats = torch.cat([Xn.real, Xn.imag, up.real, up.imag, pe],
                          dim=-1).to(torch.float32)
        h = self.phi(feats)                                               # (B, T, d_phi)
        valid = (torch.arange(T, device=X.device)[None, :]
                 < lengths[:, None]).to(h.dtype)[..., None]
        h = (h * valid).sum(1) / lengths[:, None].to(h.dtype)             # masked mean
        logp = torch.log(rms).to(torch.float32)[:, None]
        return torch.cat([h, logp], dim=-1)
