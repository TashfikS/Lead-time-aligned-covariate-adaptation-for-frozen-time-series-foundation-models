"""Synthetic leading-indicator tasks (LeadBench-S).

Every task returns, on a common target clock t = 0..C+H-1:
  y      (C+H,)  target; only y[:C] is given to models, y[C:] is the truth.
  x      (C+H,)  raw covariate; only x[:C] is observable (past-only covariate).
  xo     (C+H,)  oracle-aligned covariate xo_t = x_{t - tau_t} on the source clock;
                 known for every t with t - tau_t <= C-1, i.e. over the whole horizon when tau_t >= H.
  tau    (C+H,)  true lead time per target time (int).
The generator simulates on a longer clock with a burn-in, so all xo values used are defined.
Seeds: numpy SeedSequence((FAMILY_CODE, tau0, delta, split_code, index)) -> disjoint across splits.
"""
import numpy as np

FAMILIES = ["linear", "periodic", "pulse", "nonlinear", "tvlag", "null"]
FAM_CODE = {f: i + 1 for i, f in enumerate(FAMILIES)}
SPLIT_CODE = {"dev": 1, "test": 2, "train": 3, "val": 4}
PERIOD = 24


def _ar1(rng, n, phi, sd=1.0):
    e = rng.standard_normal(n + 200) * sd
    z = np.empty(n + 200)
    z[0] = e[0]
    for t in range(1, n + 200):
        z[t] = phi * z[t - 1] + e[t]
    z = z[200:]
    return (z - z.mean()) / (z.std() + 1e-8)


def _season(rng, n, amp):
    ph = rng.uniform(0, 2 * np.pi)
    t = np.arange(n)
    s = np.sin(2 * np.pi * t / PERIOD + ph) + 0.5 * rng.uniform(-1, 1) * np.sin(4 * np.pi * t / PERIOD + ph)
    return amp * s


def _gamma_kernel(rng, kmax=48):
    k = rng.uniform(2.0, 4.0)          # shape
    peak = rng.uniform(3.0, 8.0)       # mode in steps
    theta = peak / (k - 1)
    t = np.arange(kmax, dtype=float)
    h = np.where(t > 0, t ** (k - 1) * np.exp(-t / theta), 0.0)
    return h / h.sum()


def make_task(family, tau0, split, index, C=512, H=24, delta=None, snr=None):
    """Return dict(y, x, xo, tau, meta). tau0: nominal lead (int). delta: tvlag amplitude."""
    ss = np.random.SeedSequence((FAM_CODE[family], int(tau0), int(delta or 0), SPLIT_CODE[split], int(index)))
    rng = np.random.default_rng(ss)
    B = 400                                  # burn-in on the source clock
    n = C + H
    N = n + B + 200
    snr = float(rng.uniform(1.0, 4.0)) if snr is None else float(snr)
    meta = {"family": family, "tau0": int(tau0), "split": split, "index": int(index), "snr": snr}
    # ---- covariate on the source clock -------------------------------------------------------
    if family in ("linear", "nonlinear", "tvlag", "null"):
        phi = rng.uniform(0.85, 0.98)
        xs = _ar1(rng, N, phi)
        meta["phi_x"] = float(phi)
    elif family == "periodic":
        phi = rng.uniform(0.5, 0.9)
        a = rng.uniform(1.0, 2.0)
        xs = a * _season(rng, N, 1.0) + _ar1(rng, N, phi)
        xs = (xs - xs.mean()) / xs.std()
        meta["phi_x"] = float(phi); meta["per_amp"] = float(a)
    elif family == "pulse":
        p = rng.uniform(0.02, 0.05)
        xs = (rng.random(N) < p) * rng.exponential(1.0, N)
        xs = xs / (xs.std() + 1e-8)
        meta["p_pulse"] = float(p)
    else:
        raise ValueError(family)
    # ---- lead time per target time ------------------------------------------------------------
    t_all = np.arange(N)
    if family == "tvlag":
        d = int(delta if delta is not None else 16)
        ema = np.empty(N); ema[0] = xs[0]
        for t in range(1, N):
            ema[t] = 0.97 * ema[t - 1] + 0.03 * xs[t]
        ema = (ema - ema.mean()) / (ema.std() + 1e-8)
        st = np.concatenate([np.zeros(tau0), ema[:-tau0]])          # state observed tau0 steps earlier
        tau = np.round(tau0 - d / (1 + np.exp(-2.0 * st))).astype(int)   # higher state -> shorter lead
        meta["delta"] = d
    else:
        tau = np.full(N, int(tau0))
    src = t_all - tau
    valid = src >= 0
    aligned = np.where(valid, xs[np.clip(src, 0, None)], np.nan)
    # ---- response --------------------------------------------------------------------------------
    if family == "pulse":
        h = _gamma_kernel(rng)
        filt = np.convolve(np.nan_to_num(aligned), h)[:N]
        drive = filt / (filt[B:].std() + 1e-8)
        meta["kernel_peak"] = float(np.argmax(h))
    elif family == "nonlinear":
        kind = str(rng.choice(["tanh", "relu", "square"]))
        a = np.nan_to_num(aligned)
        if kind == "tanh":
            drive = np.tanh(2.0 * a)
        elif kind == "relu":
            drive = np.maximum(a - 0.5, 0.0)
        else:
            drive = a ** 2
        drive = (drive - drive[B:].mean()) / (drive[B:].std() + 1e-8)
        meta["nl"] = kind
    else:
        drive = np.nan_to_num(aligned)
    beta = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
    if family == "null":
        beta = 0.0                                   # covariate carries no information about the target
    amp_s = rng.uniform(0.0, 1.0)
    phie = rng.uniform(0.3, 0.8)
    noise = _ar1(rng, N, phie) / np.sqrt(snr)
    level = rng.uniform(-2, 2)
    ys = level + beta * drive + _season(rng, N, amp_s) + noise
    meta.update({"beta": float(beta), "amp_season_y": float(amp_s), "phi_e": float(phie)})
    # ---- cut to the target clock -----------------------------------------------------------------
    sl = slice(B, B + n)
    y = ys[sl].astype(np.float32)
    x = xs[sl].astype(np.float32)
    xo = aligned[sl].astype(np.float32)
    tau_c = tau[sl].astype(np.int32)
    assert np.isfinite(xo).all(), "burn-in too short for this lead"
    return {"y": y, "x": x, "xo": xo, "tau": tau_c, "meta": meta}


GATE_CELLS = [  # (family, tau0, delta)
    ("linear", 24, None), ("linear", 48, None),
    ("periodic", 30, None), ("periodic", 41, None),
    ("pulse", 24, None), ("pulse", 36, None),
    ("nonlinear", 24, None), ("nonlinear", 48, None),
    ("tvlag", 48, 12), ("tvlag", 48, 20),
]


def shift_series(x_ctx, lag, C, H):
    """Shift an observed covariate (length C) forward by integer lag onto the target clock (length C+H).
    out_t = x_{t-lag}; NaN where t-lag < 0 or t-lag >= C (unknown future)."""
    out = np.full(C + H, np.nan, dtype=np.float32)
    t = np.arange(C + H)
    s = t - int(lag)
    ok = (s >= 0) & (s < C)
    out[ok] = x_ctx[s[ok]]
    return out


def ccf_lag(y_ctx, x_ctx, kmin=1, kmax=168, prewhiten=False):
    """Classical lag estimate: argmax_k |corr(y_t, x_{t-k})| on the context. Optional AR(1) prewhitening."""
    y = np.asarray(y_ctx, float); x = np.asarray(x_ctx, float)
    if prewhiten:
        def pw(z):
            z = z - z.mean()
            a = np.dot(z[1:], z[:-1]) / (np.dot(z[:-1], z[:-1]) + 1e-12)
            return z[1:] - a * z[:-1]
        y, x = pw(y), pw(x)
    y = (y - y.mean()) / (y.std() + 1e-12)
    x = (x - x.mean()) / (x.std() + 1e-12)
    n = len(y)
    best, bk = -1.0, kmin
    for k in range(kmin, min(kmax, n - 16) + 1):
        c = abs(np.mean(y[k:] * x[: n - k]))
        if c > best:
            best, bk = c, k
    return bk, best
