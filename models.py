"""
All embedding and model architectures for cv-sbi-sim.

SBI summary networks:
  WaveformEmbedding            : full 28-ch CNN → embed_dim (64)
  ReducedWaveformEmbedding     : 4-ch CNN + scalar prefix tokens → embed_dim (64)

Autoencoder / surrogate components:
  AutoencoderEncoder                : 24-ch CNN → latent_dim
  ReducedAutoencoderEncoder         : 4-ch CNN + scalar prefix tokens → latent_dim
  LipschitzReducedAutoencoderEncoder: same + soft spectral-norm ceiling per layer
  VAEReducedAutoencoderEncoder      : stochastic variant of ReducedAutoencoderEncoder
  WaveformDecoder                   : latent_dim → MLP → out_channels*T

Surrogate:
  SurrogateDecoder                  : N_PARAMS → MLP → N_CONT*T (forward model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

from dataset import N_CHANNELS, N_CONT, N_REDUCED_CHANNELS, N_SCALARS, N_PARAMS, T

EMBED_DIM  = 64
LATENT_DIM = 128


# ── SBI summary networks ──────────────────────────────────────────────────────

class WaveformEmbedding(nn.Module):
    """Full 28-ch 1D CNN → embed_dim. Pooling: 'attention' or 'mean'."""

    CONV_LAYERS = [
        (N_CHANNELS, 64,  7, 1),
        (64,         128, 5, 2),
        (128,        256, 5, 2),
        (256,        256, 3, 1),
    ]

    def __init__(self, embed_dim=EMBED_DIM, pooling="attention"):
        super().__init__()
        self.n_channels = N_CHANNELS
        self.t          = T
        self.embed_dim  = embed_dim
        self.pooling    = pooling

        layers = []
        for in_ch, out_ch, k, s in self.CONV_LAYERS:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, stride=s), nn.SiLU()]
        self.cnn = nn.Sequential(*layers)
        if pooling == "attention":
            self.attn_pool = nn.Linear(self.CONV_LAYERS[-1][1], 1)
        self.proj = nn.Linear(self.CONV_LAYERS[-1][1], embed_dim)

    def describe(self):
        return {
            "type": "WaveformEmbedding",
            "input": f"({self.n_channels}, {self.t})",
            "pooling": self.pooling,
            "embed_dim": self.embed_dim,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, x):
        x = x.view(-1, self.n_channels, self.t)
        h = self.cnn(x).transpose(1, 2)
        if self.pooling == "attention":
            w = self.attn_pool(h).softmax(dim=1)
            h = (w * h).sum(dim=1)
        else:
            h = h.mean(dim=1)
        return self.proj(h)


class ReducedWaveformEmbedding(nn.Module):
    """4-ch pressure CNN + 5 scalars as prefix tokens → attention pool → embed_dim."""

    CONV_LAYERS = [
        (N_REDUCED_CHANNELS, 64,  7, 1),
        (64,                 128, 5, 2),
        (128,                256, 5, 2),
        (256,                256, 3, 1),
    ]

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.wave_len  = N_REDUCED_CHANNELS * T
        feat_dim = self.CONV_LAYERS[-1][1]

        layers = []
        for in_ch, out_ch, k, s in self.CONV_LAYERS:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, stride=s), nn.SiLU()]
        self.cnn          = nn.Sequential(*layers)
        self.scalar_projs = nn.ModuleList([nn.Linear(1, feat_dim) for _ in range(N_SCALARS)])
        self.attn_pool    = nn.Linear(feat_dim, 1)
        self.proj         = nn.Linear(feat_dim, embed_dim)

    @property
    def output_dim(self):
        return self.embed_dim

    def describe(self):
        return {
            "type": "ReducedWaveformEmbedding",
            "input_waveforms": f"({N_REDUCED_CHANNELS}, {T})",
            "input_scalars": "Pas_mean, Pas_max, Pas_min, SV, HR_z",
            "embed_dim": self.embed_dim,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, x):
        waves   = x[:, :self.wave_len].view(-1, N_REDUCED_CHANNELS, T)
        scalars = x[:, self.wave_len:]
        h = self.cnn(waves).transpose(1, 2)
        scalar_tokens = torch.stack(
            [proj(scalars[:, i:i+1]) for i, proj in enumerate(self.scalar_projs)],
            dim=1,
        )
        h = torch.cat([scalar_tokens, h], dim=1)
        w = self.attn_pool(h).softmax(dim=1)
        return self.proj((w * h).sum(dim=1))


# ── Autoencoder components ────────────────────────────────────────────────────

class AutoencoderEncoder(nn.Module):
    """24-ch continuous CNN → latent_dim."""

    CONV_LAYERS = [
        (N_CONT, 64,  7, 1),
        (64,     128, 5, 2),
        (128,    256, 5, 2),
        (256,    256, 3, 1),
    ]

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        feat_dim = self.CONV_LAYERS[-1][1]

        layers = []
        for in_ch, out_ch, k, s in self.CONV_LAYERS:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, stride=s), nn.SiLU()]
        self.cnn       = nn.Sequential(*layers)
        self.attn_pool = nn.Linear(feat_dim, 1)
        self.proj      = nn.Linear(feat_dim, latent_dim)

    def describe(self):
        return {
            "type": "AutoencoderEncoder",
            "input": f"({N_CONT}, {T})",
            "latent_dim": self.latent_dim,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, x):
        h = self.cnn(x.view(-1, N_CONT, T)).transpose(1, 2)
        w = self.attn_pool(h).softmax(dim=1)
        return self.proj((w * h).sum(dim=1))


class ReducedAutoencoderEncoder(nn.Module):
    """4-ch pressure CNN + 5 scalar prefix tokens → latent_dim."""

    CONV_LAYERS = [
        (N_REDUCED_CHANNELS, 64,  7, 1),
        (64,                 128, 5, 2),
        (128,                256, 5, 2),
        (256,                256, 3, 1),
    ]

    def __init__(self, latent_dim=LATENT_DIM, proj_hidden=None, n_scalars=N_SCALARS):
        super().__init__()
        self.latent_dim  = latent_dim
        self.proj_hidden = proj_hidden
        self.n_scalars   = n_scalars
        self.wave_len    = N_REDUCED_CHANNELS * T
        feat_dim = self.CONV_LAYERS[-1][1]

        layers = []
        for in_ch, out_ch, k, s in self.CONV_LAYERS:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, stride=s), nn.SiLU()]
        self.cnn          = nn.Sequential(*layers)
        self.scalar_projs = nn.ModuleList([nn.Linear(1, feat_dim) for _ in range(n_scalars)])
        self.attn_pool    = nn.Linear(feat_dim, 1)

        if proj_hidden is not None:
            self.proj  = nn.Linear(feat_dim, proj_hidden)
            self.proj2 = nn.Linear(proj_hidden, latent_dim)
        else:
            self.proj  = nn.Linear(feat_dim, latent_dim)
            self.proj2 = None

    @property
    def output_dim(self):
        return self.latent_dim

    def describe(self):
        scalars_str = ("Pas_mean, Pas_max, Pas_min, HR_z" if self.n_scalars == 4
                       else "Pas_mean, Pas_max, Pas_min, SV, HR_z")
        return {
            "type": "ReducedAutoencoderEncoder",
            "input_waveforms": f"({N_REDUCED_CHANNELS}, {T})",
            "input_scalars": scalars_str,
            "latent_dim": self.latent_dim,
            "proj_hidden": self.proj_hidden,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, x):
        waves   = x[:, :self.wave_len].view(-1, N_REDUCED_CHANNELS, T)
        scalars = x[:, self.wave_len:]
        h = self.cnn(waves).transpose(1, 2)
        scalar_tokens = torch.stack(
            [proj(scalars[:, i:i+1]) for i, proj in enumerate(self.scalar_projs)],
            dim=1,
        )
        h = torch.cat([scalar_tokens, h], dim=1)
        w = self.attn_pool(h).softmax(dim=1)
        h = self.proj((w * h).sum(dim=1))
        if self.proj2 is not None:
            h = F.silu(h)
            h = self.proj2(h)
        return h


class _SoftSpectralCeiling(nn.Module):
    """Weight parametrization: clamp spectral norm to at most `ceiling`."""

    def __init__(self, ceiling: float, n_power_iters: int = 1):
        super().__init__()
        self.ceiling       = ceiling
        self.n_power_iters = n_power_iters
        self.register_buffer('_u', None)
        self.register_buffer('_v', None)
        self.last_sigma: float = 0.0

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        W_mat = W.reshape(W.shape[0], -1)
        h, w  = W_mat.shape

        if self._u is None or self._u.shape[0] != h:
            self._u = F.normalize(W_mat.new_empty(h).normal_(), dim=0)
        if self._v is None or self._v.shape[0] != w:
            self._v = F.normalize(W_mat.new_empty(w).normal_(), dim=0)

        u, v = self._u.detach(), self._v.detach()
        with torch.no_grad():
            for _ in range(self.n_power_iters):
                v = F.normalize(W_mat.t() @ u, dim=0, eps=1e-12)
                u = F.normalize(W_mat @ v,     dim=0, eps=1e-12)

        sigma = (u @ (W_mat @ v)).abs()
        if self.training:
            self._u.copy_(u)
            self._v.copy_(v)

        scale = (self.ceiling / sigma.clamp(min=1e-12)).clamp(max=1.0)
        self.last_sigma = (sigma * scale).item()
        return W * scale

    def right_inverse(self, W: torch.Tensor) -> torch.Tensor:
        return W


def _sn(module: nn.Module, ceiling: float) -> nn.Module:
    P.register_parametrization(module, 'weight', _SoftSpectralCeiling(ceiling))
    return module


class LipschitzReducedAutoencoderEncoder(nn.Module):
    """ReducedAutoencoderEncoder with soft spectral-norm ceiling on every weight matrix."""

    CONV_LAYERS = ReducedAutoencoderEncoder.CONV_LAYERS

    def __init__(self, latent_dim: int = LATENT_DIM, sn_ceiling: float = 2.0, proj_hidden=None):
        super().__init__()
        self.latent_dim  = latent_dim
        self.sn_ceiling  = sn_ceiling
        self.proj_hidden = proj_hidden
        self.wave_len    = N_REDUCED_CHANNELS * T
        feat_dim = self.CONV_LAYERS[-1][1]

        layers = []
        for in_ch, out_ch, k, s in self.CONV_LAYERS:
            layers += [
                _sn(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, stride=s), sn_ceiling),
                nn.SiLU(),
            ]
        self.cnn          = nn.Sequential(*layers)
        self.scalar_projs = nn.ModuleList(
            [_sn(nn.Linear(1, feat_dim), sn_ceiling) for _ in range(N_SCALARS)]
        )
        self.attn_pool    = _sn(nn.Linear(feat_dim, 1), sn_ceiling)

        if proj_hidden is not None:
            self.proj  = _sn(nn.Linear(feat_dim, proj_hidden), sn_ceiling)
            self.proj2 = _sn(nn.Linear(proj_hidden, latent_dim), sn_ceiling)
        else:
            self.proj  = _sn(nn.Linear(feat_dim, latent_dim), sn_ceiling)
            self.proj2 = None

    @property
    def output_dim(self):
        return self.latent_dim

    def spectral_norms(self) -> dict[str, float]:
        return {
            name: mod.last_sigma
            for name, mod in self.named_modules()
            if isinstance(mod, _SoftSpectralCeiling)
        }

    def describe(self):
        return {
            "type": "LipschitzReducedAutoencoderEncoder",
            "input_waveforms": f"({N_REDUCED_CHANNELS}, {T})",
            "input_scalars": "Pas_mean, Pas_max, Pas_min, SV, HR_z",
            "latent_dim": self.latent_dim,
            "proj_hidden": self.proj_hidden,
            "sn_ceiling": self.sn_ceiling,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        waves   = x[:, :self.wave_len].view(-1, N_REDUCED_CHANNELS, T)
        scalars = x[:, self.wave_len:]
        h = self.cnn(waves).transpose(1, 2)
        scalar_tokens = torch.stack(
            [proj(scalars[:, i:i+1]) for i, proj in enumerate(self.scalar_projs)],
            dim=1,
        )
        h = torch.cat([scalar_tokens, h], dim=1)
        w = self.attn_pool(h).softmax(dim=1)
        h = self.proj((w * h).sum(dim=1))
        if self.proj2 is not None:
            h = F.silu(h)
            h = self.proj2(h)
        return h


class WaveformDecoder(nn.Module):
    """MLP decoder: latent_dim → hidden → out_channels*T.

    out_channels: N_CONT (24) for continuous waveforms only.
    Output shape: (B, out_channels*T) — flat.
    """

    def __init__(self, latent_dim=LATENT_DIM, hidden=512, n_layers=2, out_channels=N_CONT):
        super().__init__()
        self.out_channels = out_channels
        out_dim = out_channels * T
        layers = [nn.Linear(latent_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def describe(self):
        return {
            "type": "WaveformDecoder",
            "latent_dim": self.net[0].in_features,
            "hidden": self.net[0].out_features,
            "n_layers": sum(1 for m in self.net if isinstance(m, nn.Linear)),
            "out_channels": self.out_channels,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, z):
        return self.net(z)


class VAEReducedAutoencoderEncoder(nn.Module):
    """Stochastic (VAE-style) variant of ReducedAutoencoderEncoder.

    forward() returns (z, mu, log_var).
    Training: z = mu + eps * exp(0.5 * log_var)  [reparameterization]
    Eval:     z = mu,  log_var = zeros
    """

    CONV_LAYERS = ReducedAutoencoderEncoder.CONV_LAYERS

    def __init__(self, latent_dim: int = LATENT_DIM, proj_hidden: int = None):
        super().__init__()
        self.latent_dim  = latent_dim
        self.proj_hidden = proj_hidden
        self.wave_len    = N_REDUCED_CHANNELS * T
        feat_dim = self.CONV_LAYERS[-1][1]

        layers = []
        for in_ch, out_ch, k, s in self.CONV_LAYERS:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, stride=s), nn.SiLU()]
        self.cnn          = nn.Sequential(*layers)
        self.scalar_projs = nn.ModuleList([nn.Linear(1, feat_dim) for _ in range(N_SCALARS)])
        self.attn_pool    = nn.Linear(feat_dim, 1)

        proj_in = feat_dim
        if proj_hidden is not None:
            self.proj = nn.Linear(feat_dim, proj_hidden)
            proj_in   = proj_hidden
        else:
            self.proj = None

        self.fc_mu      = nn.Linear(proj_in, latent_dim)
        self.fc_log_var = nn.Linear(proj_in, latent_dim)

    @property
    def output_dim(self):
        return self.latent_dim

    def describe(self):
        return {
            "type": "VAEReducedAutoencoderEncoder",
            "input_waveforms": f"({N_REDUCED_CHANNELS}, {T})",
            "input_scalars": "Pas_mean, Pas_max, Pas_min, SV, HR_z",
            "latent_dim": self.latent_dim,
            "proj_hidden": self.proj_hidden,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        waves   = x[:, :self.wave_len].view(-1, N_REDUCED_CHANNELS, T)
        scalars = x[:, self.wave_len:]
        h = self.cnn(waves).transpose(1, 2)
        scalar_tokens = torch.stack(
            [proj(scalars[:, i:i+1]) for i, proj in enumerate(self.scalar_projs)],
            dim=1,
        )
        h = torch.cat([scalar_tokens, h], dim=1)
        w = self.attn_pool(h).softmax(dim=1)
        return (w * h).sum(dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self._features(x)
        if self.proj is not None:
            h = F.silu(self.proj(h))
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        if self.training:
            z = mu + (0.5 * log_var).exp() * torch.randn_like(mu)
        else:
            z       = mu
            log_var = torch.zeros_like(mu)
        return z, mu, log_var


# ── PasHR summary network ─────────────────────────────────────────────────────

class PasHREncoder(nn.Module):
    """1-ch Pas waveform CNN + HR scalar → latent_dim.

    Mirrors ReducedAutoencoderEncoder but with 1 waveform channel and 1 scalar.
    Used as summary network for Flow A (minimal cath input).
    Input x: (B, T+1) — z-scored Pas waveform (T=201) concatenated with z-scored HR.
    """

    CONV_LAYERS = [
        (1,   64,  7, 1),
        (64,  128, 5, 2),
        (128, 256, 5, 2),
        (256, 256, 3, 1),
    ]

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        feat_dim = self.CONV_LAYERS[-1][1]

        layers = []
        for in_ch, out_ch, k, s in self.CONV_LAYERS:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, stride=s), nn.SiLU()]
        self.cnn      = nn.Sequential(*layers)
        self.hr_proj  = nn.Linear(1, feat_dim)
        self.attn_pool = nn.Linear(feat_dim, 1)
        self.proj      = nn.Linear(feat_dim, latent_dim)

    @property
    def output_dim(self):
        return self.latent_dim

    def describe(self):
        return {
            "type": "PasHREncoder",
            "input": f"Pas waveform ({T},) + HR scalar",
            "latent_dim": self.latent_dim,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pas  = x[:, :-1].unsqueeze(1)          # (B, 1, T)
        hr   = x[:, -1:]                        # (B, 1)
        h    = self.cnn(pas).transpose(1, 2)    # (B, T', feat_dim)
        hr_t = self.hr_proj(hr).unsqueeze(1)    # (B, 1, feat_dim)
        h    = torch.cat([hr_t, h], dim=1)      # (B, T'+1, feat_dim)
        w    = self.attn_pool(h).softmax(dim=1)
        return self.proj((w * h).sum(dim=1))


# ── Inverse encoder ───────────────────────────────────────────────────────────

class InverseEncoder(nn.Module):
    """24-ch continuous CNN → z-scored θ (N_PARAMS).

    Paired with a frozen SurrogateDecoder to form a waveform→θ→waveform
    autoencoder. Training loss is reconstruction MSE on output waveforms;
    per-parameter R² at eval time reveals which parameters are identifiable
    from the full set of 24 sim waveforms (theoretical ceiling).

    Input:  flat z-scored waveforms (N_CONT * T,)
    Output: z-scored theta (N_PARAMS,) — use theta_norm.json to convert to raw units
    """

    CONV_LAYERS = AutoencoderEncoder.CONV_LAYERS

    def __init__(self, hidden_theta: int = 256):
        super().__init__()
        self.hidden_theta = hidden_theta
        feat_dim = self.CONV_LAYERS[-1][1]

        layers = []
        for in_ch, out_ch, k, s in self.CONV_LAYERS:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, stride=s), nn.SiLU()]
        self.cnn       = nn.Sequential(*layers)
        self.attn_pool = nn.Linear(feat_dim, 1)
        self.proj      = nn.Sequential(
            nn.Linear(feat_dim, hidden_theta),
            nn.SiLU(),
            nn.Linear(hidden_theta, N_PARAMS),
        )

    def describe(self):
        return {
            "type": "InverseEncoder",
            "input": f"({N_CONT}, {T})",
            "output_dim": N_PARAMS,
            "hidden_theta": self.hidden_theta,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x.view(-1, N_CONT, T)).transpose(1, 2)  # (B, T', feat_dim)
        w = self.attn_pool(h).softmax(dim=1)
        return self.proj((w * h).sum(dim=1))                  # (B, N_PARAMS)


# ── Surrogate ─────────────────────────────────────────────────────────────────

class SurrogateDecoder(nn.Module):
    """Forward surrogate model: θ ∈ ℝ^N_PARAMS → 24 continuous waveforms × T.

    Output shape: (B, N_CONT * T) — flat. Reshape to (B, N_CONT, T) as needed.
    """

    def __init__(self, hidden: int = 512, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(N_PARAMS, hidden), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, N_CONT * T))
        self.net = nn.Sequential(*layers)

    def describe(self):
        return {
            "type": "SurrogateDecoder",
            "input_dim": N_PARAMS,
            "hidden": self.net[0].out_features,
            "n_layers": sum(1 for m in self.net if isinstance(m, nn.Linear)),
            "output": f"({N_CONT}, {T})",
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self.net(theta)
