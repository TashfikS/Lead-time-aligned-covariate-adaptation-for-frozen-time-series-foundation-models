"""Classical lead-time alignment baselines. Each returns aligned covariate channels on the target clock
(k_out, C+H) with NaN where the source is unobserved; the evaluation harness appends the raw covariate."""
import zlib
import numpy as np
from .synth import shift_series, ccf_lag


def gcc_phat_lag(y, x, kmin=0, kmax=168):
    """Generalized cross-correlation with phase transform (Knapp & Carter, 1976). Lag l: y_t ~ x_{t-l}."""
    y = np.asarray(y, float) - np.mean(y); x = np.asarray(x, float) - np.mean(x)
    n = len(y); L = 1 << (2 * n - 1).bit_length()
    Y = np.fft.rfft(y, L); X = np.fft.rfft(x, L)
    G = Y * np.conj(X)
    r = np.fft.irfft(G / (np.abs(G) + 1e-12), L)
    seg = np.abs(r[kmin:kmax + 1])
    return int(kmin + np.argmax(seg))


def topk_ccf_lags(y, x, k=3, kmin=0, kmax=168, min_sep=3):
    y = (np.asarray(y, float) - np.mean(y)) / (np.std(y) + 1e-12)
    x = (np.asarray(x, float) - np.mean(x)) / (np.std(x) + 1e-12)
    n = len(y)
    c = np.array([abs(np.mean(y[l:] * x[: n - l])) if l < n - 16 else 0.0 for l in range(kmin, kmax + 1)])
    order = np.argsort(-c); out = []
    for i in order:
        l = kmin + int(i)
        if all(abs(l - o) >= min_sep for o in out):
            out.append(l)
        if len(out) == k:
            break
    return out


def align_classical(method, y_ctx, x_ctx, C, H, kmax=168, global_lag=None, seed_key=None):
    """y_ctx (C,), x_ctx (C,) -> (channels (m, C+H), lags list)."""
    if method == "ccf":
        lags = [ccf_lag(y_ctx, x_ctx, 0, kmax)[0]]
    elif method == "ccf_pw":
        lags = [ccf_lag(y_ctx, x_ctx, 0, kmax, prewhiten=True)[0]]
    elif method == "gcc_phat":
        lags = [gcc_phat_lag(y_ctx, x_ctx, 0, kmax)]
    elif method == "top3":
        lags = topk_ccf_lags(y_ctx, x_ctx, 3, 0, kmax)
    elif method == "global":
        lags = [int(global_lag)]
    elif method == "randlag":
        r = np.random.default_rng(np.random.SeedSequence((777, zlib.crc32(str(seed_key).encode()))))
        lags = [int(r.integers(H, kmax + 1))]
    else:
        raise ValueError(method)
    return np.stack([shift_series(x_ctx, l, C, H) for l in lags]), lags
