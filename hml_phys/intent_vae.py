"""Intent VAE (MIND §4.2; docs/07 §20).

MIND encodes humanoid STATE sequences (no actions) into a compact latent with "a 1D causal convolutional
architecture [Xiao et al. 2025]" trained with "the same loss function as sigma-VAE [Rybkin et al. 2021]",
temporal downsampling 4, latent dim 32, lambda_KL = 1e-5; once trained the VAE is frozen and only the
encoder is kept. Xiao et al. 2025 is MotionStreamer; this module is its causal temporal autoencoder
(vendor_motionstreamer @ 8aace3f: models/causal_cnn.py, models/resnet.py, models/tae.py, utils/losses.py)
ported line for line, with only the input width changed (272 -> our 366-d state) and latent 16 -> 32.

State = the non-action part of our physics token, already normalised with TokenStats:
    root token 15 (trans 3, rot6d 6, lin-vel 3, ang-vel 3) + body state 351
    (local_positions 72, local_vel 72, dof_pose_6d 138, dof_vel 69)                    = 366 channels
It contains everything in MIND's 358-d state (root height, root-frame joint positions, 6D rotations,
linear/angular velocities) plus the window-canonical root xy / yaw.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hml_phys.tokens import ROOT_DIM, STATE_DIM as BODY_STATE_DIM

VAE_STATE_DIM = ROOT_DIM + BODY_STATE_DIM   # 366


# ---------------------------------------------------------------- MotionStreamer models/resnet.py (causal part)
class CausalResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation="relu"):
        super().__init__()
        self.activation1 = nn.ReLU() if activation == "relu" else nn.SiLU()
        self.activation2 = nn.ReLU() if activation == "relu" else nn.SiLU()
        self.left_padding = (3 - 1) * dilation
        self.conv1 = nn.Conv1d(n_in, n_state, kernel_size=3, stride=1, padding=0, dilation=dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_orig = x
        x = self.activation1(x)                      # norm=None in MotionStreamer's configuration
        x = F.pad(x, (self.left_padding, 0))
        x = self.conv1(x)
        x = self.activation2(x)
        x = self.conv2(x)
        return x + x_orig


class CausalResnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation="relu"):
        super().__init__()
        blocks = [CausalResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** d, activation=activation)
                  for d in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------- MotionStreamer models/causal_cnn.py
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation + (1 - stride)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))    # left padding only


class CausalEncoder(nn.Module):
    def __init__(self, input_emb_width, hidden_size=1024, down_t=2, stride_t=2, width=1024, depth=3,
                 dilation_growth_rate=3, activation="relu", latent_dim=32, clip_range=(-30.0, 20.0)):
        super().__init__()
        self.clip_range = clip_range
        self.proj = nn.Linear(width, latent_dim * 2)
        filter_t = stride_t * 2
        blocks = [CausalConv1d(input_emb_width, width, 3, 1, 1), nn.ReLU()]
        for _ in range(down_t):
            blocks.append(nn.Sequential(CausalConv1d(width, width, filter_t, stride_t, 1),
                                        CausalResnet1D(width, depth, dilation_growth_rate, activation=activation)))
        blocks.append(CausalConv1d(width, hidden_size, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):                            # x [B, C, T]
        x = self.model(x).transpose(1, 2)
        mu, logvar = self.proj(x).chunk(2, dim=2)
        logvar = torch.clamp(logvar, self.clip_range[0], self.clip_range[1])
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return z, mu, logvar


class CausalDecoder(nn.Module):
    def __init__(self, input_emb_width, hidden_size=1024, down_t=2, stride_t=2, width=1024, depth=3,
                 dilation_growth_rate=3, activation="relu"):
        super().__init__()
        blocks = [CausalConv1d(hidden_size, width, 3, 1, 1), nn.ReLU()]
        for _ in range(down_t):
            blocks.append(nn.Sequential(
                CausalResnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation),
                nn.Upsample(scale_factor=stride_t, mode="nearest"),
                CausalConv1d(width, width, 3, 1, 1)))
        blocks += [CausalConv1d(width, width, 3, 1, 1), nn.ReLU(), CausalConv1d(width, input_emb_width, 3, 1, 1)]
        self.model = nn.Sequential(*blocks)

    def forward(self, z):                            # z [B, T', hidden]
        return self.model(z.transpose(1, 2))


# ---------------------------------------------------------------- MotionStreamer models/tae.py (Causal_TAE)
class IntentVAE(nn.Module):
    def __init__(self, input_dim=VAE_STATE_DIM, hidden_size=1024, width=1024, down_t=2, stride_t=2, depth=3,
                 dilation_growth_rate=3, activation="relu", latent_dim=32, clip_range=(-30.0, 20.0)):
        super().__init__()
        self.input_dim, self.latent_dim, self.down = input_dim, latent_dim, stride_t ** down_t
        self.decode_proj = nn.Linear(latent_dim, width)
        self.encoder = CausalEncoder(input_dim, hidden_size, down_t, stride_t, width, depth, dilation_growth_rate,
                                     activation=activation, latent_dim=latent_dim, clip_range=clip_range)
        self.decoder = CausalDecoder(input_dim, hidden_size, down_t, stride_t, width, depth, dilation_growth_rate,
                                     activation=activation)

    def encode(self, x):
        """x [B, T, 366] (normalised state) -> z, mu, logvar each [B, T/4, latent]."""
        return self.encoder(x.permute(0, 2, 1).float())

    def decode(self, z):
        """z [B, T/4, latent] -> [B, T, 366]."""
        return self.decoder(self.decode_proj(z)).permute(0, 2, 1)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        return self.decode(z), mu, logvar

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------- MotionStreamer utils/losses.py (ReConsLoss)
def _softclip(t, lo):
    return lo + F.softplus(t - lo)


def sigma_vae_nll(pred, gt, dims=None):
    """Optimal-sigma VAE reconstruction loss (Rybkin et al. 2021), exactly MotionStreamer's ReConsLoss.forward:
    one shared sigma = RMSE over (batch, time, channel), soft-clipped at log sigma >= -6, Gaussian NLL SUMMED."""
    if dims is not None:
        pred, gt = pred[..., dims], gt[..., dims]
    log_sigma = ((gt - pred) ** 2).mean([0, 1, 2], keepdim=True).sqrt().log()
    log_sigma = _softclip(log_sigma, -6)
    nll = 0.5 * torch.pow((gt - pred) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)
    return nll.sum()


def kl_loss(mu, logvar):
    """MotionStreamer's forward_KL: summed over latent dims and time, averaged over the batch."""
    return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1, 2))).mean()


def load_intent_vae(path, device="cuda"):
    ck = torch.load(path, map_location="cpu")
    a = ck["args"]
    vae = IntentVAE(input_dim=int(a.get("input_dim", VAE_STATE_DIM)), hidden_size=a["width"], width=a["width"],
                    down_t=a["down_t"], depth=a["depth"], dilation_growth_rate=a["dilation"], latent_dim=a["latent_dim"])
    vae.load_state_dict(ck["model"]); vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, ck
