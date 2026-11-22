import sys, os, json, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from lta import river as RV, supervised as SV
from lta.arms import mase_windows
DSN = sys.argv[1] if len(sys.argv) > 1 else "river"
CFGD = {"river": ("data/river", RV.SPLITS, {}), "traffic": ("data/traffic", RV.TRAFFIC_SPLITS, dict(event_kind="delta", event_delta=20.0, splits=RV.TRAFFIC_SPLITS, step_minutes=5))}[DSN]
Hd = RV.load_hourly(CFGD[0] + "/hourly.csv"); pairs = json.load(open(CFGD[0] + "/pairs.json")); rows = []; T0 = time.time()
for name, (up, dn) in pairs.items():
    idx, U, D = SV.filled_pair(Hd, up, dn); tz = idx.tz
    T = {k: tuple(pd.Timestamp(v, tz=tz) for v in CFGD[1][k]) for k in ("train", "val", "test")}
    trm = (idx >= T["train"][0]) & (idx < T["train"][1])
    fu, fd = 1e-3 * np.nanmedian(U[trm]), 1e-3 * np.nanmedian(D[trm])     # floor for non-positive discharge (D21)
    nclip = int(np.sum(U <= 0) + np.sum(D <= 0))
    lU, lD = np.log(np.where(np.isnan(U), np.nan, np.maximum(U, fu))), np.log(np.where(np.isnan(D), np.nan, np.maximum(D, fd)))
    WTs = RV.make_windows(Hd, {name: pairs[name]}, "test", **CFGD[2])
    pos = pd.Series(np.arange(len(idx)), index=idx); pte = np.array([int(pos[pd.Timestamp(w["meta"]["origin"])]) for w in WTs])
    ptr = SV.valid_positions(idx, U, D, *T["train"]); pva = SV.valid_positions(idx, U, D, *T["val"])
    Xtr, Ytr, _ = SV.design(idx, lU, lD, ptr); Xva, Yva, _ = SV.design(idx, lU, lD, pva); Xte, _, bte = SV.design(idx, lU, lD, pte)
    f, _ = SV.fit_lgbm(Xtr, Ytr, Xva, Yva)
    pred = np.exp(f(Xte) + bte[:, None]); ms = mase_windows(pred, WTs, 512, 24)
    rows += [{"set": DSN + "_test", "arm": "lgbm_full_log", "cell": name, "id": w["meta"]["origin"], "event": w["meta"]["event"], "mase": float(ms[i])} for i, w in enumerate(WTs)]
    print(f"[{(time.time()-T0)/60:.1f}] {name} clipped={nclip} GM={np.exp(np.log(ms).mean()):.4f}", flush=True)
    pd.DataFrame(rows).to_csv(f"results/supervised/{DSN}_log_rows.csv", index=False)
pd.DataFrame(rows).to_csv(f"results/supervised/{DSN}_log_rows.csv", index=False); print("DONE")
