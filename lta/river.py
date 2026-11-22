"""RiverLead: upstream -> downstream discharge forecasting windows from USGS hourly data.

Each window dict mirrors the synthetic task format: y (C+H,) downstream flow, x (C+H,) upstream flow
(only x[:C] is used by any method), meta (pair, origin, split, event flag)."""
import numpy as np
import pandas as pd

SPLITS = {"train": ("2015-10-01", "2021-10-01"), "val": ("2021-10-01", "2022-10-01"),
          "test": ("2022-10-01", "2025-10-01")}


def load_hourly(path="data/river/hourly.csv"):
    return pd.read_csv(path, index_col=0, parse_dates=True)


TRAFFIC_SPLITS = {"train": ("2017-01-01", "2017-04-01"), "val": ("2017-04-01", "2017-05-01"),
                  "test": ("2017-05-01", "2017-07-01")}
ENERGY_SPLITS = {"train": ("2015-01-01", "2018-01-01"), "val": ("2018-01-01", "2019-01-01"),
                 "test": ("2019-01-01", "2020-10-01")}


def make_windows(Hd, pairs, split, C=512, H=24, stride=24, max_gap=6, event_ratio=1.5, event_kind="ratio",
                 event_delta=8.0, splits=None, event_frac=0.5, step_minutes=60):
    """event_kind: "ratio" (rivers), "delta" (AirLead, absolute 24-h range), "relrange" (EnergyLead: 24-h range of
    the upwind series >= event_frac x its train-split 99th percentile)."""
    splits = splits or SPLITS
    a, b = splits[split]
    out = []
    for name, (up, dn) in pairs.items():
        d = Hd[[up, dn]].copy()
        filled = d.interpolate(limit=max_gap, limit_area="inside")
        idx = filled.index
        t0 = pd.Timestamp(a, tz=idx.tz); t1 = pd.Timestamp(b, tz=idx.tz)
        if step_minutes == 60:                 # hourly benchmarks: daily origins at 00 (unchanged)
            origins = idx[(idx >= t0) & (idx < t1) & (idx.hour == 0)][::max(1, stride // 24)]
        else:                                  # sub-hourly (TrafficLead): an origin every `stride` steps from midnight
            every = pd.Timedelta(minutes=step_minutes * stride)
            origins = idx[(idx >= t0) & (idx < t1) & (((idx - idx.normalize()) % every) == pd.Timedelta(0))]
        U, D = filled[up].values, filled[dn].values
        if event_kind == "relrange":
            ta, tb = (pd.Timestamp(v, tz=idx.tz) for v in splits["train"])
            thr = event_frac * float(np.nanpercentile(U[(idx >= ta) & (idx < tb)], 99))
        pos = pd.Series(np.arange(len(idx)), index=idx)
        for o in origins:
            p = int(pos[o])                     # first horizon step = origin
            if p - C < 0 or p + H > len(idx):
                continue
            y = D[p - C:p + H]; x = U[p - C:p + H]
            if not (np.isfinite(y).all() and np.isfinite(x).all()):
                continue
            if (idx[p + H - 1] - idx[p - C]) != pd.Timedelta(minutes=step_minutes * (C + H - 1)):
                continue
            if event_kind == "ratio":
                u48 = x[C - 48:C]
                event = bool(u48.max() >= event_ratio * max(u48.min(), 1e-6))
            elif event_kind == "relrange":
                u24 = x[C - 24:C]
                event = bool(u24.max() - u24.min() >= thr)
            else:                                           # range over the last 24 context steps
                u24 = x[C - 24:C]
                event = bool(u24.max() - u24.min() >= event_delta)
            out.append({"y": y.astype(np.float32), "x": x.astype(np.float32),
                        "meta": {"pair": name, "origin": str(o), "split": split, "event": event, "index": len(out)},
                        "cell": name})
    return out
