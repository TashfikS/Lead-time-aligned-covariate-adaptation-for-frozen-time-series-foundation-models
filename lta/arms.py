"""Arm construction shared by real-data benchmarks. Every aligned arm returns [aligned channels..., raw] so the
backbone always also sees the raw past covariate (alignment can only add information)."""
import numpy as np
from .baselines import align_classical
from .synth import shift_series


def _logdiff(z):
    return np.diff(np.log(np.clip(z, 1e-3, None)))


def _ccf_peak(y, x, kmax):
    y = (y - y.mean()) / (y.std() + 1e-9); x = (x - x.mean()) / (x.std() + 1e-9); n = len(y)
    c = np.array([np.mean(y[l:] * x[: n - l]) for l in range(kmax + 1)])
    return int(np.argmax(c)), float(c.max())


def fit_global_lag(train_windows, C, kmax=168):
    """One lag per group (pair), estimated on concatenated train-window log-differences."""
    out = {}
    groups = sorted({w["cell"] for w in train_windows})
    for g in groups:
        ws = [w for w in train_windows if w["cell"] == g]
        num = np.zeros(kmax + 1); cnt = 0
        for w in ws[::7]:                       # non-overlapping (512 h ≈ 21 days; every 7th daily origin)
            y = _logdiff(w["y"][:C]); x = _logdiff(w["x"][:C])
            if y.std() < 1e-6 or x.std() < 1e-6:
                continue
            y = (y - y.mean()) / y.std(); x = (x - x.mean()) / x.std(); n = len(y)
            num += np.array([np.mean(y[l:] * x[: n - l]) for l in range(kmax + 1)]); cnt += 1
        out[g] = int(np.argmax(num / max(cnt, 1)))
    return out


def fit_binned_lag(train_windows, C, kmax=168, nbins=5, min_corr=0.3):
    """State-dependent classical lag: windowed CCF lag on log-diffs vs upstream-flow quintile (per group)."""
    out = {}
    for g in sorted({w["cell"] for w in train_windows}):
        ws = [w for w in train_windows if w["cell"] == g][::7]
        lags, flows, all_l, all_f = [], [], [], []
        for w in ws:
            y = _logdiff(w["y"][:C]); x = _logdiff(w["x"][:C])
            if y.std() < 1e-6 or x.std() < 1e-6:
                continue
            l, c = _ccf_peak(y, x, kmax)
            all_l.append(l); all_f.append(float(np.median(w["x"][:C])))
            if c >= min_corr:
                lags.append(l); flows.append(float(np.median(w["x"][:C])))
        if not lags:                     # no window passes the threshold (never happened on RiverLead/AirLead)
            lags, flows = all_l, all_f
        lags, flows = np.array(lags), np.array(flows)
        edges = np.quantile(flows, np.linspace(0, 1, nbins + 1)[1:-1]) if len(flows) >= nbins else np.array([])
        b = np.digitize(flows, edges)
        med = [int(np.median(lags[b == i])) if (b == i).any() else int(np.median(lags)) for i in range(nbins)]
        out[g] = {"edges": edges.tolist(), "lags": med, "n": int(len(lags))}
    return out


def arm_covariates(windows, arm, C, H, kmax=168, global_lags=None, binned=None, lta_aligned=None):
    """-> (y_ctx (N,C), cov_ctx (N,k,C) | None, cov_fut (N,k,H) | None, lags list)."""
    N = len(windows)
    y = np.stack([w["y"][:C] for w in windows]).astype(np.float32)
    if arm == "uni":
        return y, None, None, None
    raw = np.stack([w["x"][:C] for w in windows]).astype(np.float32)
    if arm == "raw":
        return y, raw[:, None], np.full((N, 1, H), np.nan, np.float32), None
    chans, lags = [], []
    for i, w in enumerate(windows):
        xc = w["x"][:C]
        if arm in ("ccf", "ccf_pw", "gcc_phat", "top3"):
            ch, lg = align_classical(arm, w["y"][:C], xc, C, H, kmax)
        elif arm == "global":
            ch, lg = align_classical("global", None, xc, C, H, kmax, global_lag=global_lags[w["cell"]])
        elif arm == "binned":
            bd = binned[w["cell"]]
            q = float(np.mean(xc[-24:]))
            lg0 = bd["lags"][int(np.digitize([q], bd["edges"])[0])]
            ch, lg = align_classical("global", None, xc, C, H, kmax, global_lag=lg0)
        elif arm == "randlag":
            ch, lg = align_classical("randlag", None, xc, C, H, kmax,
                                     seed_key=(w["cell"], w["meta"].get("origin", w["meta"].get("index"))))
        elif arm == "lta":
            ch, lg = lta_aligned[i], [-1]
        else:
            raise ValueError(arm)
        chans.append(ch); lags.append(lg)
    al = np.stack(chans).astype(np.float32)                          # (N,m,C+H)
    cc = np.concatenate([al[:, :, :C], raw[:, None]], 1)
    cf = np.concatenate([al[:, :, C:], np.full((N, 1, H), np.nan, np.float32)], 1)
    return y, cc, cf, lags


def mase_windows(pred, windows, C, H, m=24):
    out = np.zeros(len(windows))
    for i, w in enumerate(windows):
        y = w["y"]
        s = np.mean(np.abs(y[m:C] - y[:C - m])) + 1e-8
        out[i] = np.mean(np.abs(pred[i] - y[C:C + H])) / s
    return out
