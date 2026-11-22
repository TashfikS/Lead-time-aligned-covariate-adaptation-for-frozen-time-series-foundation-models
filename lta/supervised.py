"""P5-C supervised lagged baselines (ARX ridge, LightGBM), direct multi-horizon, per pair.

Origin convention matches lta.river.make_windows: for a window with origin position p the context is
[p-C, p) and the horizon is [p, p+H). Features use only values at positions < p.
"""
import numpy as np
import pandas as pd

Y_LAGS = list(range(1, 49)) + [72, 96, 120, 144, 168]
X_LAGS = list(range(1, 169))
KMAX = 168


def filled_pair(Hd, up, dn, max_gap=6):
    d = Hd[[up, dn]].interpolate(limit=max_gap, limit_area="inside")
    return d.index, d[up].values.astype(np.float64), d[dn].values.astype(np.float64)


def design(idx, U, D, pos, H=24, use_cov=True):
    """Feature matrix and targets for origin positions `pos` (array of int). Returns X, Yt (n,H), base (n,)."""
    pos = np.asarray(pos)
    base = D[pos - 1]
    F = [base[:, None]]
    F.append(np.stack([D[pos - k] - base for k in Y_LAGS if k > 1], 1))
    if use_cov:
        F.append(np.stack([U[pos - k] for k in X_LAGS], 1))
    hr = idx[pos].hour.values; dw = idx[pos].dayofweek.values
    F.append(np.stack([np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)], 1))
    F.append(np.eye(7)[dw])
    X = np.concatenate(F, 1)
    hz = np.arange(H)
    Yt = np.stack([D[pos + h] for h in hz], 1) - base[:, None] if (pos + H <= len(D)).all() else None
    return X, Yt, base


def valid_positions(idx, U, D, t0, t1, H=24, step=1):
    n = len(D)
    p = np.arange(KMAX, n - H)
    ok = (idx[p] >= t0) & (idx[p] < t1)
    p = p[ok][::step]
    keep = []
    for q in p:
        if np.isfinite(D[q - KMAX:q + H]).all() and np.isfinite(U[q - KMAX:q]).all():
            keep.append(q)
    return np.array(keep, dtype=int)


class Std:
    def fit(self, X):
        self.m = X.mean(0); self.s = X.std(0) + 1e-9; return self

    def __call__(self, X):
        return (X - self.m) / self.s


def fit_ridge(Xtr, Ytr, Xva, Yva, alphas=(0.1, 1, 10, 100, 1000)):
    sx = Std().fit(Xtr); sy = Std().fit(Ytr)
    A, B = sx(Xtr), sy(Ytr); Av, Bv = sx(Xva), sy(Yva)
    A1 = np.c_[A, np.ones(len(A))]; Av1 = np.c_[Av, np.ones(len(Av))]
    G = A1.T @ A1; R = A1.T @ B
    best = None
    for a in alphas:
        reg = a * np.eye(G.shape[0]); reg[-1, -1] = 0.0
        W = np.linalg.solve(G + reg, R)
        e = np.mean((Av1 @ W - Bv) ** 2)
        if best is None or e < best[0]:
            best = (e, a, W)
    _, a, W = best
    return lambda X: sy.s * (np.c_[sx(X), np.ones(len(X))] @ W) + sy.m, {"alpha": a}


def fit_lgbm(Xtr, Ytr, Xva, Yva, H=24, n_jobs=4):
    import lightgbm as lgb
    models, iters = [], []
    params = dict(objective="l2", num_leaves=31, learning_rate=0.05, min_data_in_leaf=50, feature_fraction=0.8,
                  bagging_seed=0, feature_fraction_seed=0, seed=0, verbose=-1, num_threads=n_jobs)
    for h in range(H):
        dtr = lgb.Dataset(Xtr, Ytr[:, h]); dva = lgb.Dataset(Xva, Yva[:, h], reference=dtr)
        b = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        models.append(b); iters.append(b.best_iteration)
    return lambda X: np.stack([m.predict(X, num_iteration=m.best_iteration) for m in models], 1), {"iters": iters}
