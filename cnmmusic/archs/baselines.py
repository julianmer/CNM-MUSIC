####################################################################################################
#                                           baselines.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/07/26                                                                                #
#                                                                                                  #
# Purpose: Learned DoA baselines ported to this framework: SubspaceNet (Shmuel, Merkofer et al.,   #
#          TVT 2024; tau-lag autocorrelation CNN -> surrogate covariance K^H K + eps I),           #
#          DA-MUSIC (Merkofer et al., TVT 2023; GRU -> surrogate covariance -> nominal-manifold    #
#          spectrum -> MLP peak finder), and GridCNN (Papageorgiou, Sellathurai & Eldar, TSP       #
#          2021; 3-channel covariance -> CNN -> sigmoid grid classification).                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import math

import torch
import torch.nn as nn


#**************************************************************************************************#
#                                       Class AntiRectifier                                        #
#**************************************************************************************************#
class AntiRectifier(nn.Module):
    def forward(self, x):
        return torch.cat([torch.relu(x), torch.relu(-x)], dim=1)


#**************************************************************************************************#
#                                        Class SubspaceNet                                         #
#**************************************************************************************************#
#                                                                                                  #
# tau lagged autocorrelation features [B, tau, 2N, N] -> conv/deconv autoencoder -> complex K ->   #
# surrogate covariance Rz = K^H K + eps I, consumed by any (differentiable) subspace method.       #
#                                                                                                  #
#**************************************************************************************************#
class SubspaceNet(nn.Module):
    def __init__(self, N=8, tau=8, eps=1.0):
        super().__init__()
        self.N, self.tau, self.eps = N, tau, eps
        self.conv1 = nn.Conv2d(tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(64, 16, kernel_size=2)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=2)
        self.anti = AntiRectifier()
        self.drop = nn.Dropout(0.2)

    def pre_process(self, X):
        """Complex snapshots (B, N, T) -> tau lagged autocorrelations (B, tau, 2N, N), float."""
        X = X - X.mean(dim=-1, keepdim=True)
        T = X.shape[-1]
        feats = []
        for i in range(self.tau):
            Ri = X[..., :T - i] @ X[..., i:].mH / (T - i)
            feats.append(torch.cat([Ri.real, Ri.imag], dim=-2))
        return torch.stack(feats, dim=1).to(torch.float32)

    def forward(self, X):
        h = self.pre_process(X)
        h = self.anti(self.conv1(h))
        h = self.anti(self.conv2(h))
        h = self.anti(self.conv3(h))
        h = self.anti(self.deconv2(h))
        h = self.anti(self.deconv3(h))
        h = self.deconv4(self.drop(h))[:, 0]                        # (B, 2N, N)
        K = torch.complex(h[:, :self.N], h[:, self.N:]).to(torch.complex128)
        eye = torch.eye(self.N, dtype=K.dtype, device=K.device)
        return K.mH @ K + self.eps * eye                            # (B, N, N) PSD surrogate


#**************************************************************************************************#
#                                          Class DAMUSIC                                           #
#**************************************************************************************************#
#                                                                                                  #
# Faithful to create_model_alternative of the reference (github.com/julianmer/DA-MUSIC_ICASSP22):  #
# BN -> GRU(2N) -> complex surrogate K -> eig, sorted by |eigenvalue| descending -> HARD noise-    #
# subspace slice evecs[:, :, d:] (= the m-d smallest, the reference's intent) -> spectrum          #
# Dense(2N)+ReLU -> output angles (N-1 wide, sliced to d: variable-d adaptation). The steering     #
# vectors of the ACTUAL aperture are an EXPLICIT forward input (batch data, like n_src), built     #
# by the caller from the batch's sensor positions on self.grid (ULA-identical for a ULA). The      #
# source count d is always known (the unknown-d protocol is not used in this project).             #
#                                                                                                  #
#**************************************************************************************************#
class DAMUSIC(nn.Module):
    def __init__(self, N=8, grid_size=361, theta_range=(-torch.pi / 2, torch.pi / 2), **kwargs):
        super().__init__()
        self.N = N
        # per-channel batch statistics over (batch, time), any T -- the faithful PyTorch
        # equivalent of the original Keras BatchNormalization(axis=-1) on (B, T, 2N)
        self.norm = nn.BatchNorm1d(2 * N)
        self.gru = nn.GRU(2 * N, 2 * N, batch_first=True)
        self.fc_k = nn.Linear(2 * N, 2 * N * N)
        grid = torch.linspace(*theta_range, grid_size, dtype=torch.float64)
        self.fc1 = nn.Linear(grid_size, 2 * N)                      # three DISTINCT Dense(2N)
        self.fc2 = nn.Linear(2 * N, 2 * N)                          # layers, no dropout, per
        self.fc2b = nn.Linear(2 * N, 2 * N)                         # the reference
        self.fc3 = nn.Linear(2 * N, N - 1)                          # d_max = N - 1 angles
        self.register_buffer('grid', grid)

    def forward(self, X, n_src, A_grid):
        """X (B, N, T) snapshots, n_src known count, A_grid (N, G) aperture steering vectors."""
        h = torch.cat([X.real, X.imag], dim=1).to(torch.float32)    # (B, 2N, T)
        if self.training and h.shape[0] * h.shape[-1] < 2:
            # a single (B=1, T=1) group: batch statistics are undefined -- use running stats
            h = nn.functional.batch_norm(h, self.norm.running_mean, self.norm.running_var,
                                         self.norm.weight, self.norm.bias, False,
                                         self.norm.momentum, self.norm.eps)
        else:
            h = self.norm(h)
        h = h.transpose(1, 2)                                       # (B, T, 2N)
        _, hn = self.gru(h)
        K = self.fc_k(hn[0])
        K = torch.complex(K[:, :self.N * self.N], K[:, self.N * self.N:]
                          ).reshape(-1, self.N, self.N).to(torch.complex128)
        evals, evecs = torch.linalg.eig(K)
        # sort |eigenvalue| descending, so [:, :, d:] = the m-d SMALLEST = the noise subspace
        # (the reference comment's intent; its code relied on a learned order with fixed d)
        order = evals.abs().argsort(dim=-1, descending=True)
        En = torch.gather(evecs, -1, order[:, None, :].expand_as(evecs))[:, :, n_src:]
        A_grid = A_grid.to(En.dtype)
        q = (torch.einsum('bmk,bmg->bkg', En.conj(), A_grid) if A_grid.ndim == 3      # per-scene
             else torch.einsum('bmk,mg->bkg', En.conj(), A_grid)).abs().pow(2).sum(1)
        P = (1.0 / (q + 1e-12)).to(torch.float32)                   # MUSIC spectrum, (B, G)
        h = torch.relu(self.fc1(P))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc2b(h))
        return self.fc3(h).to(torch.float64)                        # (B, N-1), first d used


#**************************************************************************************************#
#                                          Class GridCNN                                           #
#**************************************************************************************************#
#                                                                                                  #
# 3-channel covariance image (Re, Im, angle) -> 4 conv layers (256 ch) -> FC stack -> sigmoid      #
# multi-label grid classification (distinct conv layers 2-4, per the paper; the reference repo     #
# reuses one layer's weights).                                                                     #
#                                                                                                  #
#**************************************************************************************************#
class GridCNN(nn.Module):
    def __init__(self, N=8, grid_size=121, theta_range=(-1.05, 1.05)):
        super().__init__()
        self.N = N
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=2)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=2)
        self.bn1, self.bn2 = nn.BatchNorm2d(256), nn.BatchNorm2d(256)
        self.bn3, self.bn4 = nn.BatchNorm2d(256), nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * (N - 5) * (N - 5), 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc_angle = nn.Linear(1024, grid_size)
        self.drop = nn.Dropout(0.2)          # 20%, per the paper
        self.register_buffer('grid', torch.linspace(*theta_range, grid_size,
                                                    dtype=torch.float64))

    def pre_process(self, X, R=None):
        """3-channel covariance image; R = None uses the sample covariance of X."""
        if R is None:
            R = X @ X.mH / X.shape[-1]
        return torch.stack([R.real, R.imag, R.angle()], dim=1).to(torch.float32)

    def forward(self, X, R=None):
        h = self.pre_process(X, R)
        h = torch.relu(self.bn1(self.conv1(h)))
        h = torch.relu(self.bn2(self.conv2(h)))
        h = torch.relu(self.bn3(self.conv3(h)))
        h = torch.relu(self.bn4(self.conv4(h)))
        h = h.flatten(1)
        h = self.drop(torch.relu(self.fc1(h)))
        h = self.drop(torch.relu(self.fc2(h)))
        h = self.drop(torch.relu(self.fc3(h)))
        return self.fc_angle(h)                                     # (B, G) logits

    def decode(self, logits, n_src):
        """Top-n_src local maxima of the sigmoid grid -> DoAs (B, n_src)."""
        p = torch.sigmoid(logits)
        interior = (p[:, 1:-1] >= p[:, :-2]) & (p[:, 1:-1] >= p[:, 2:])
        vals = torch.where(interior, p[:, 1:-1], torch.full_like(p[:, 1:-1], -1.0))
        top_v, top_i = vals.topk(n_src, dim=-1)
        idx = torch.where(top_v > -1.0, top_i + 1, p.topk(n_src, dim=-1).indices)
        return self.grid[idx].sort(dim=-1).values

    def targets(self, doas):
        """Multi-hot grid targets from true DoAs (B, d) -> (B, G)."""
        t = torch.zeros(doas.shape[0], len(self.grid), device=doas.device)
        idx = (doas[..., None] - self.grid[None, None, :]).abs().argmin(-1)
        return t.scatter(1, idx, 1.0)
