"""Diebold-Mariano test with the Harvey-Leybourne-Newbold (1997) small-sample correction, per pair.

Loss per window: log MASE (so the differential is the log ratio used everywhere else in the paper).
Windows within a pair are ordered by origin. Their horizons do not overlap (origins are >= H steps apart), so
each window is one forecast; residual serial dependence between consecutive windows is handled by a
Newey-West long-run variance with bandwidth floor(n^(1/3)). HLN correction with h = 1; Student-t, n - 1 df.
"""
import numpy as np
from scipy import stats


def dm_hln(d):
    d = np.asarray(d, float); n = len(d); m = d.mean(); e = d - m
    L = int(np.floor(n ** (1 / 3)))
    v = np.dot(e, e) / n
    for k in range(1, L + 1):
        v += 2 * (1 - k / (L + 1)) * np.dot(e[k:], e[:-k]) / n
    dm = m / np.sqrt(v / n)
    hln = dm * np.sqrt((n - 1) / n)                    # h = 1: sqrt((n + 1 - 2h + h(h-1)/n) / n)
    return float(hln), float(2 * stats.t.sf(abs(hln), n - 1)), float(m)


def dm_pairs(wide, a, b, alpha=0.05):
    """wide: DataFrame indexed by (cell, origin) with columns of per-window MASE. Returns per-pair results and counts."""
    out = {}
    for cell, g in wide.groupby(level=0):
        g = g.sort_index(level=1)
        stat, p, m = dm_hln(np.log(g[a].values) - np.log(g[b].values))
        out[cell] = (stat, p, m)
    better = sum(1 for s, p, m in out.values() if p < alpha and m < 0)
    worse = sum(1 for s, p, m in out.values() if p < alpha and m > 0)
    return out, better, worse
