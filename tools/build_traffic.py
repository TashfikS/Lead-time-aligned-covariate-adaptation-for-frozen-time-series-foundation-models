"""Rebuild data/traffic/hourly.csv (TrafficLead) from the public PEMS-BAY speed file.

The CSV is not redistributed in this repository (Caltrans PeMS terms). Download `pems-bay.h5`
(Li et al., ICLR 2018; e.g. from the DCRNN data release) and run:

    python tools/build_traffic.py path/to/pems-bay.h5

Output: data/traffic/hourly.csv with a `time` column (5-minute steps, 2017-01-01 to 2017-06-30)
and one speed column (mph, 2 decimals; missing readings, coded 0 in PEMS-BAY, left blank) per detector used in data/traffic/pairs.json, sorted by id.
(The file name follows the other benchmarks; TrafficLead itself is 5-minute data.)
A `pems_bay.npz` export (keys speed, t, sensors) is also accepted as input.
"""
import json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    if path.endswith(".npz"):
        z = np.load(path)
        return pd.DataFrame(z["speed"], index=pd.to_datetime(z["t"]), columns=[str(s) for s in z["sensors"]])
    df = pd.read_hdf(path)
    df.columns = [str(c) for c in df.columns]
    return df


def main(src, pairs=os.path.join(ROOT, "data/traffic/pairs.json"), out=os.path.join(ROOT, "data/traffic/hourly.csv")):
    df = load(src)
    ids = sorted({s for p in json.load(open(pairs)).values() for s in p})
    missing = [s for s in ids if s not in df.columns]
    if missing:
        sys.exit(f"detectors missing from {src}: {missing}")
    sub = df[ids].astype(np.float32).replace(0.0, np.nan)   # PEMS-BAY codes missing readings as 0
    sub.index = pd.DatetimeIndex(sub.index).strftime("%Y-%m-%d %H:%M:%S")
    sub.index.name = "time"
    sub.to_csv(out, float_format="%.2f")
    print(f"wrote {out}: {sub.shape[0]} rows x {sub.shape[1]} detectors")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:])
