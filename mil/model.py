"""MIL v1 (Section 5 of the handoff): per-clip CNN encoder with global max pooling and gated attention pooling
(Ilse et al. 2018). `pool` applies the attention head to precomputed clip embeddings; it is exactly the
second half of `forward`, used for fast evaluation of overlapping windows (the encoder is per clip)."""

import torch
from torch import nn


class ClipEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))

        self.convs = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, emb_dim))

    def forward(self, x):  # (N, 1, 64, 626)
        return self.convs(x).amax(dim=(2, 3))  # global max over frequency and time


def freq_mixstyle(x, p=0.4, alpha=0.3, eps=1e-6):
    """Freq-MixStyle (CPJKU DCASE baseline): with probability p per batch, re-normalise each clip's per-frequency
    statistics (over channel and time) and replace them with a Beta(alpha, alpha) mix of its own and a random
    other clip's statistics. Training only; x: (N, C, F, T)."""
    if torch.rand(1).item() > p:
        return x
    mu = x.mean(dim=(1, 3), keepdim=True).detach()
    sig = (x.var(dim=(1, 3), keepdim=True) + eps).sqrt().detach()
    lam = torch.distributions.Beta(alpha, alpha).sample((x.shape[0], 1, 1, 1)).to(x.device, x.dtype)
    perm = torch.randperm(x.shape[0], device=x.device)
    return (x - mu) / sig * (lam * sig + (1 - lam) * sig[perm]) + lam * mu + (1 - lam) * mu[perm]


class MfccEncoder(nn.Module):
    """Trainable clip encoder for small whole-clip MFCC maps (1, 13, T), T = 15 or 30. Input is standardised per
    coefficient with training-fold statistics (buffers). Two 2x2 max pools (13xT -> 3x(T//4)), then global max."""

    def __init__(self, n_coef=13, emb_dim=128):
        super().__init__()

        def conv(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU()]

        self.register_buffer("mu", torch.zeros(1, 1, n_coef, 1))
        self.register_buffer("sd", torch.ones(1, 1, n_coef, 1))
        self.convs = nn.Sequential(*conv(1, 32), *conv(32, 64), nn.MaxPool2d(2), *conv(64, 128), nn.MaxPool2d(2),
                                   *conv(128, emb_dim))

    def forward(self, x):  # (N, 1, 13, T)
        return self.convs((x - self.mu) / self.sd).amax(dim=(2, 3))


class AttentionMIL(nn.Module):
    def __init__(self, emb_dim=128, att_dim=64, dropout=0.3):
        super().__init__()
        self.encoder = ClipEncoder(emb_dim)
        self.att_V, self.att_U = nn.Linear(emb_dim, att_dim), nn.Linear(emb_dim, att_dim)
        self.att_w = nn.Linear(att_dim, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(emb_dim, 1))

    def pool(self, h):  # (B, K, emb) -> logit (B, 1), attention (B, K)
        a = torch.softmax(self.att_w(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h))), dim=1)
        return self.classifier((a * h).sum(dim=1)), a.squeeze(-1)

    def forward(self, bags):  # (B, K, 1, 64, 626)
        B, K = bags.shape[:2]
        h = self.encoder(bags.flatten(0, 1)).view(B, K, -1)
        return self.pool(h)


class GatedAttention(nn.Module):
    def __init__(self, dim, att_dim=64):
        super().__init__()
        self.V, self.U, self.w = nn.Linear(dim, att_dim), nn.Linear(dim, att_dim), nn.Linear(att_dim, 1)

    def forward(self, h):  # (B, K, dim) -> weights (B, K, 1)
        return torch.softmax(self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))), dim=1)


class MILModel(nn.Module):
    """Generalised MIL v1 for the run list.

    input: "cnn" (trainable ClipEncoder on log-mel), "mfcc13x<T>" (trainable MfccEncoder on whole-clip MFCC)
    or "emb_<name>" (frozen embeddings of size in_dim,
    standardised with training-fold mean/sd stored as buffers). pooling: "gated" (B0), "heads4" (4 independent
    gated heads, concatenated) or "transformer" (one TransformerEncoderLayer with a learned positional embedding
    before a gated head). classifier: "linear" (dropout 0.3 + linear, B0) or "mlp" (dim -> 128 -> 1, ReLU,
    dropout 0.3)."""

    def __init__(self, input="cnn", in_dim=128, pooling="gated", classifier="linear", k=24, mixstyle=False):
        super().__init__()
        self.mixstyle = mixstyle
        self.is_mfcc = input.startswith("mfcc")
        self.is_cnn = input == "cnn" or self.is_mfcc  # trainable conv encoder on a (1, F, T) map
        dim = 128 if self.is_cnn else in_dim
        self.out_dim = dim
        if self.is_mfcc:
            self.encoder_net = MfccEncoder(13, 128)
        elif self.is_cnn:
            self.encoder_net = ClipEncoder(128)
        else:
            self.register_buffer("in_mu", torch.zeros(dim))
            self.register_buffer("in_sd", torch.ones(dim))
        self.transformer = None
        if pooling == "transformer":
            self.pos = nn.Parameter(torch.randn(1, k, dim) * 0.02)
            self.transformer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=256, dropout=0.1,
                                                          batch_first=True)
        self.heads = nn.ModuleList([GatedAttention(dim) for _ in range(4 if pooling == "heads4" else 1)])
        pooled = dim * len(self.heads)
        if classifier == "mlp":
            self.classifier = nn.Sequential(nn.Linear(pooled, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
        else:
            self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(pooled, 1))

    def encoder(self, x):  # cnn: (N, 1, 64, 626) -> (N, 128); emb: (N, D) -> standardised (N, D)
        return self.encoder_net(x) if self.is_cnn else (x - self.in_mu) / self.in_sd

    def pool(self, h):
        if self.transformer is not None:
            h = self.transformer(h + self.pos[:, :h.shape[1]])
        att = [head(h) for head in self.heads]
        z = torch.cat([(a * h).sum(dim=1) for a in att], dim=-1)
        return self.classifier(z), att[0].squeeze(-1)

    def forward(self, bags):  # cnn: (B, K, 1, 64, 626); emb: (B, K, D)
        B, K = bags.shape[:2]
        x = bags.flatten(0, 1)
        if self.training and self.mixstyle and self.is_cnn:
            x = freq_mixstyle(x)
        h = self.encoder(x).view(B, K, -1)
        return self.pool(h)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print(n_params(AttentionMIL()))
