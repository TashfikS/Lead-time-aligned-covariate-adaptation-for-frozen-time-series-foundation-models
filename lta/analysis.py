"""Pre-registered river statistics (P2): GM over windows within pair, GM over pairs; 7-day block bootstrap over
calendar weeks (shared across pairs); Wilcoxon over per-pair GM ratios."""
import numpy as np
import pandas as pd
from scipy import stats


def wide(df, metric="mase"):
    """rows (arm, cell, id) -> wide frame index (cell, id), columns arms."""
    return df.pivot_table(index=["cell", "id"], columns="arm", values=metric, aggfunc="first")


def gm(v):
    v = np.asarray(v, float)
    return float(np.exp(np.mean(np.log(v))))


def pair_ratios(w, arm, ref):
    d = w[[arm, ref]].dropna()
    lr = np.log(d[arm]) - np.log(d[ref])
    return lr.groupby(level=0).mean()                      # per-pair mean log ratio


def pooled_ratio(w, arm, ref):
    return float(np.exp(pair_ratios(w, arm, ref).mean()))


def block_bootstrap(w, arm, ref, B=2000, block_days=7, seed=0):
    """CI for pooled GM ratio; resample calendar blocks of origins jointly across pairs."""
    d = w[[arm, ref]].dropna().copy()
    lr = (np.log(d[arm]) - np.log(d[ref])).rename("lr").reset_index()
    t = pd.to_datetime(lr["id"], utc=True)
    lr["blk"] = ((t - t.min()).dt.days // block_days).astype(int)
    blocks = np.sort(lr["blk"].unique())
    # per (pair, block): sum and count of log ratios
    agg = lr.groupby(["cell", "blk"])["lr"].agg(["sum", "count"]).reset_index()
    pairs = sorted(agg["cell"].unique())
    S = np.zeros((len(pairs), len(blocks))); N = np.zeros_like(S)
    pi = {p: i for i, p in enumerate(pairs)}; bi = {b: j for j, b in enumerate(blocks)}
    for r in agg.itertuples():
        S[pi[r.cell], bi[r.blk]] = r.sum; N[pi[r.cell], bi[r.blk]] = r.count
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(blocks), len(blocks))
        s = S[:, idx].sum(1); n = N[:, idx].sum(1)
        ok = n > 0
        out[b] = np.exp(np.mean(s[ok] / n[ok]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def compare(w, arm, ref, B=2000):
    pr = pair_ratios(w, arm, ref)
    lo, hi = block_bootstrap(w, arm, ref, B=B)
    p = stats.wilcoxon(pr.values).pvalue if len(pr) >= 5 and np.any(pr.values != 0) else np.nan
    return {"arm": arm, "ref": ref, "pooled_ratio": float(np.exp(pr.mean())), "ci_lo": lo, "ci_hi": hi,
            "pairs_better": int((pr < 0).sum()), "n_pairs": int(len(pr)), "wilcoxon_p_pairs": float(p),
            **{f"r_{k}": float(np.exp(v)) for k, v in pr.items()}}


def best_on_val(wval, candidates, ref="raw"):
    scores = {a: pooled_ratio(wval, a, ref) for a in candidates if a in wval.columns}
    return min(scores, key=scores.get), scores
