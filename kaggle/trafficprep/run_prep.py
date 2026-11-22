# ---- Export PEMS-BAY speed matrix (h5 -> compressed npz) for local use. CFG + lta/ injected. ----
import glob, numpy as np, pandas as pd
OUT = _W
f = [p for p in glob.glob("/kaggle/input/**/pems-bay.h5", recursive=True)]; print("found", f, flush=True)
df = pd.read_hdf(f[0]); print(df.shape, df.index.min(), df.index.max(), flush=True)
np.savez_compressed(os.path.join(OUT, "pems_bay.npz"), speed=df.values.astype(np.float32),
                    t=df.index.astype("int64").values, sensors=np.array([str(c) for c in df.columns]))
m = [p for p in glob.glob("/kaggle/input/**/pems-bay-meta.h5", recursive=True)]
if m:
    try:
        meta = pd.read_hdf(m[0]); meta.to_csv(os.path.join(OUT, "pems_bay_meta.csv")); print("meta", meta.shape, list(meta.columns)[:12], flush=True)
    except Exception as e:
        print("meta fail", repr(e), flush=True)
print("DONE", flush=True)
