"""Lead-Time Aligner (LTA): a small trainable module placed in front of a frozen TSFM.

Given a target context y (B,C) and past-only covariates x (B,k,C), it produces lead-time-aligned covariates
xa (B,k,C+H) on the target clock, where xa_t is a soft average of source values x_s whose predicted arrival
time s + lag_s matches t. Values whose sources are not yet observed (s > C) are masked (NaN), so a leading
covariate becomes known over the first part of the horizon and unknown elsewhere.

Components
  1. lag evidence    : five cross-correlation profiles over lags 0..K (raw, differenced, AR(1)-prewhitened,
                       |diff| magnitude, rank), computed by FFT on the context.
  2. static lag head : 1-D CNN over the lag axis -> lag logits s(l), instance embedding e, log-temperature.
  3. dynamic offset  : dtau_s = D * tanh(MLP([phi_s, e])), phi_s = local covariate state at source time s
                       (value, two EMAs, slope). Zero-initialised, so training starts from a static lag.
  4. soft alignment  : w(t,s) ∝ exp(interp(s, t - s - dtau_s) / T) for 0 <= t-s-dtau_s <= K, normalised over
                       all source positions s in [0, C+H); xa_t = sum_{s<C} w x_s / sum_{s<C} w, masked when the
                       observed share of the weight is below rho.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _standardize(z, dim=-1, eps=1e-6):
    m = z.mean(dim, keepdim=True)
    s = z.std(dim, keepdim=True)
    return (z - m) / (s + eps)


def _ccf(y, x, K):
    """y (..., n), x (..., n) standardized; returns corr(y_t, x_{t-l}) for l = 0..K, shape (..., K+1)."""
    n = y.shape[-1]
    L = 1 << (2 * n - 1).bit_length()
    Y = torch.fft.rfft(y, L)
    X = torch.fft.rfft(x, L)
    r = torch.fft.irfft(Y * X.conj(), L)[..., : K + 1]          # sum_t y_t x_{t-l}
    denom = torch.arange(n, n - K - 1, -1, device=y.device, dtype=y.dtype).clamp(min=1)
    return r / denom


def _prewhiten(z):
    z = z - z.mean(-1, keepdim=True)
    a = (z[..., 1:] * z[..., :-1]).sum(-1, keepdim=True) / ((z[..., :-1] ** 2).sum(-1, keepdim=True) + 1e-8)
    e = z[..., 1:] - a * z[..., :-1]
    return F.pad(e, (1, 0))


def _rank(z):
    return torch.argsort(torch.argsort(z, -1), -1).to(z.dtype)


def lag_evidence(y, x, K):
    """y (B,C), x (B,k,C) -> (B,k,5,K+1)."""
    yb = y[:, None, :].expand_as(x)
    feats = []
    pairs = [
        (yb, x),
        (F.pad(torch.diff(yb, dim=-1), (1, 0)), F.pad(torch.diff(x, dim=-1), (1, 0))),
        (_prewhiten(yb), _prewhiten(x)),
        (F.pad(torch.diff(yb, dim=-1).abs(), (1, 0)), F.pad(torch.diff(x, dim=-1).abs(), (1, 0))),
        (_rank(yb), _rank(x)),
    ]
    for a, b in pairs:
        feats.append(_ccf(_standardize(a), _standardize(b), K))
    return torch.stack(feats, 2)


def _ema(x, alpha):
    """x (..., n); causal EMA via scan in chunks (n is small)."""
    out = torch.empty_like(x)
    acc = x[..., 0]
    for t in range(x.shape[-1]):
        acc = alpha * acc + (1 - alpha) * x[..., t]
        out[..., t] = acc
    return out


class LeadTimeAligner(nn.Module):
    def __init__(self, K=168, D=24.0, width=32, emb=32, rho=0.5, dynamic=True, init_temp=1.0, ccf_init_scale=20.0):
        super().__init__()
        self.K, self.D, self.rho, self.dynamic = K, float(D), rho, dynamic
        c_in = 5 + 1                                               # evidence + normalised lag position
        self.cnn = nn.Sequential(
            nn.Conv1d(c_in, width, 7, padding=3), nn.GELU(),
            nn.Conv1d(width, width, 7, padding=6, dilation=2), nn.GELU(),
            nn.Conv1d(width, width, 7, padding=12, dilation=4), nn.GELU(),
        )
        self.lag_out = nn.Conv1d(width, 1, 1)
        nn.init.zeros_(self.lag_out.weight); nn.init.zeros_(self.lag_out.bias)
        # residual over the classical estimate: logits start at ccf_scale * |CCF_raw(l)|
        self.ccf_scale = nn.Parameter(torch.tensor(float(ccf_init_scale)))
        self.emb = nn.Sequential(nn.Linear(2 * width, emb), nn.GELU(), nn.Linear(emb, emb))
        self.temp = nn.Linear(emb, 1)
        nn.init.zeros_(self.temp.weight); nn.init.constant_(self.temp.bias, math.log(init_temp))
        self.dyn = nn.Sequential(nn.Linear(5 + emb, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, 1))
        nn.init.zeros_(self.dyn[-1].weight); nn.init.zeros_(self.dyn[-1].bias)

    def lag_logits(self, y, x):
        """Returns static lag logits (B,k,K+1), instance embedding (B,k,emb), temperature (B,k)."""
        B, k, C = x.shape
        ev = lag_evidence(y, x, self.K)                            # (B,k,5,K+1)
        pos = torch.linspace(0, 1, self.K + 1, device=x.device, dtype=x.dtype).expand(B, k, 1, self.K + 1)
        h = self.cnn(torch.cat([ev, pos], 2).reshape(B * k, 6, self.K + 1))   # (Bk,width,K+1)
        logits = self.ccf_scale * ev[:, :, 0].abs() + self.lag_out(h).reshape(B, k, self.K + 1)
        pooled = torch.cat([h.mean(-1), h.amax(-1)], -1)
        e = self.emb(pooled).reshape(B, k, -1)
        T = F.softplus(self.temp(e)).squeeze(-1) + 0.05
        return logits, e, T

    def offsets(self, x, e, H):
        """Per-source-time lag offset dtau (B,k,C+H); source positions >= C hold the last state."""
        B, k, C = x.shape
        xs = _standardize(x)
        # scale-free level relative to the context median (meaningful for positive series such as discharge)
        med = x.median(-1, keepdim=True).values
        lvl = torch.where(med > 0, torch.log(x.clamp(min=1e-6) / med.clamp(min=1e-6)).clamp(-5, 5), xs.clamp(-5, 5))
        phi = torch.stack([xs, _ema(xs, 0.9), _ema(xs, 0.98), F.pad(torch.diff(xs, dim=-1), (1, 0)), lvl], -1)
        phi = torch.cat([phi, phi[:, :, -1:].expand(B, k, H, 5)], 2)           # (B,k,C+H,5)
        z = torch.cat([phi, e[:, :, None, :].expand(B, k, C + H, e.shape[-1])], -1)
        return self.D * torch.tanh(self.dyn(z).squeeze(-1))

    def forward(self, y, x, H, return_aux=False):
        """y (B,C), x (B,k,C) -> xa (B,k,C+H) with NaN where not observable; aux dict for losses/diagnostics."""
        B, k, C = x.shape
        N = C + H
        logits, e, T = self.lag_logits(y, x)
        dtau = self.offsets(x, e, H) if self.dynamic else torch.zeros(B, k, N, device=x.device, dtype=x.dtype)
        xa, known, share = self.align(x, logits, T, dtau, H)
        if return_aux:
            return xa, {"logits": logits, "T": T, "dtau": dtau, "share": share, "known": known}
        return xa

    def align(self, x, logits, T, dtau, H):
        """Soft forward-splat alignment. x (B,k,C); logits (B,k,K+1); T (B,k); dtau (B,k,C+H)."""
        B, k, C = x.shape
        N = C + H
        K = self.K
        # source axis = K unobserved positions before the context, the C observed ones, H unobserved future ones
        t = torch.arange(N, device=x.device, dtype=x.dtype)
        s = torch.arange(-K, N, device=x.device, dtype=x.dtype)
        dtau = torch.cat([dtau[:, :, :1].expand(B, k, K), dtau], 2)                  # (B,k,K+N)
        x = torch.cat([torch.zeros(B, k, K, device=x.device, dtype=x.dtype), x], 2)  # observed block at [K, K+C)
        lag = t[None, None, :, None] - s[None, None, None, :] - dtau[:, :, None, :]   # (B,k,t,s)
        valid = (lag >= 0) & (lag <= self.K)
        li = lag.clamp(0, self.K)
        lo = li.floor().long().clamp(max=self.K - 1)
        fr = li - lo.to(li.dtype)
        lg = logits[:, :, None, :].expand(B, k, N, self.K + 1)
        g0 = torch.gather(lg, 3, lo)
        g1 = torch.gather(lg, 3, lo + 1)
        score = (g0 * (1 - fr) + g1 * fr) / T[:, :, None, None]
        score = score.masked_fill(~valid, -1e9)
        w = torch.softmax(score, -1)                                # over source positions s in [-K, N)
        w_obs = w[..., K:K + C]
        share = w_obs.sum(-1)                                       # observed share of the weight
        xa = (w_obs * x[:, :, None, K:K + C]).sum(-1) / share.clamp(min=1e-6)
        anyvalid = valid[..., K:K + C].any(-1)
        known = (share >= self.rho) & anyvalid
        xa = torch.where(known, xa, torch.full_like(xa, float("nan")))
        return xa, known, share
