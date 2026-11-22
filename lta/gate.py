"""Phase-0 headroom gate: arm construction and metrics (protocol: protocol/P0_gate_protocol.md)."""
import zlib
import numpy as np
from .synth import make_task, GATE_CELLS, shift_series, ccf_lag

ARMS = ["uni", "raw", "oracle", "oracle_pastonly", "ccf", "ccf_pw", "randlag"]
KMAX = 168


def build_tasks(split, n_per_cell, C, H, cells=GATE_CELLS):
    tasks = []
    for fam, tau0, d in cells:
        for i in range(n_per_cell):
            T = make_task(fam, tau0, split, i, C=C, H=H, delta=d)
            T["cell"] = f"{fam}_{tau0}" + (f"_d{d}" if d else "")
            tasks.append(T)
    return tasks


def arm_inputs(tasks, arm, C, H):
    """Return (y_ctx (N,C), cov_ctx (N,1,C) or None, cov_fut (N,1,H) or None, lags (N,) or None)."""
    N = len(tasks)
    y = np.stack([t["y"][:C] for t in tasks]).astype(np.float32)
    if arm == "uni":
        return y, None, None, None
    cc = np.zeros((N, 1, C), np.float32)
    cf = np.full((N, 1, H), np.nan, np.float32)
    lags = np.full(N, -1)
    for i, t in enumerate(tasks):
        if arm == "raw":
            cc[i, 0] = t["x"][:C]
        elif arm in ("oracle", "oracle_pastonly"):
            cc[i, 0] = t["xo"][:C]
            if arm == "oracle":
                cf[i, 0] = t["xo"][C:C + H]
            lags[i] = int(t["tau"][C])
        else:
            if arm == "ccf":
                lag, _ = ccf_lag(t["y"][:C], t["x"][:C], 1, KMAX, prewhiten=False)
            elif arm == "ccf_pw":
                lag, _ = ccf_lag(t["y"][:C], t["x"][:C], 1, KMAX, prewhiten=True)
            elif arm == "randlag":
                r = np.random.default_rng(np.random.SeedSequence((777, t["meta"]["index"], t["meta"]["tau0"],
                                                                 zlib.crc32(t["cell"].encode()))))
                lag = int(r.integers(H, KMAX + 1))
            else:
                raise ValueError(arm)
            s = shift_series(t["x"][:C], lag, C, H)
            cc[i, 0] = s[:C]
            cf[i, 0] = s[C:]
            lags[i] = lag
    return y, cc, cf, lags


def mase(pred, tasks, C, H, m=24):
    out = np.zeros(len(tasks))
    for i, t in enumerate(tasks):
        y = t["y"]
        scale = np.mean(np.abs(y[m:C] - y[:C - m])) + 1e-8
        out[i] = np.mean(np.abs(pred[i] - y[C:C + H])) / scale
    return out
