"""Rerun the TimesFM-2.5 no-covariate arm with corrected horizon slicing (D12) on the exact task sets of the
gate (dev, 50/cell), RiverLead test and AirLead test. Step 0 verifies that the old slicing reproduces the
Kaggle numbers on a subsample."""
import sys, os, json, time
sys.path.insert(0, "."); os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, pandas as pd, torch
torch.set_num_threads(4)
from lta import backbones as B, river as RV
from lta.gate import build_tasks, mase
from lta.arms import mase_windows
C, H = 512, 24
m = B.TimesFM25(model_id="models/timesfm-2.5")
def uni(Y, bs=256):
    out_new, out_old = [], []
    for s in range(0, len(Y), bs):
        pt, _ = m.model.forecast(horizon=H, inputs=[r for r in Y[s:s+bs]])
        pt = np.asarray(pt); out_new.append(pt[:, -H:]); out_old.append(pt[:, :H])
    return np.concatenate(out_new), np.concatenate(out_old)
os.makedirs("results/timesfm_uni_fix", exist_ok=True); T0 = time.time()
# gate
tasks = build_tasks("dev", 50, C, H)
Y = np.stack([t["y"][:C] for t in tasks]).astype(np.float32)
pn, po = uni(Y)
g = pd.DataFrame({"cell": [t["cell"] for t in tasks], "index": [t["meta"]["index"] for t in tasks],
                  "mase_new": mase(pn, tasks, C, H), "mase_old": mase(po, tasks, C, H)})
g.to_csv("results/timesfm_uni_fix/gate_uni.csv", index=False)
print("gate", len(g), "GM old", round(float(np.exp(np.log(g.mase_old).mean())), 4), "new", round(float(np.exp(np.log(g.mase_new).mean())), 4), f"{(time.time()-T0)/60:.1f} min", flush=True)
# river + air
for name, path, pairs_p, ek in [("river", "data/river/hourly.csv", "data/river/pairs.json", {}),
                                ("air", "data/weather/hourly.csv", "data/weather/pairs.json", {"event_kind": "delta", "event_delta": 8.0})]:
    Hd = RV.load_hourly(path); pairs = json.load(open(pairs_p))
    ws = RV.make_windows(Hd, pairs, "test", C=C, H=H, **ek)
    Y = np.stack([w["y"][:C] for w in ws]).astype(np.float32)
    pn, po = uni(Y)
    d = pd.DataFrame({"cell": [w["cell"] for w in ws], "id": [w["meta"]["origin"] for w in ws], "event": [w["meta"]["event"] for w in ws],
                      "mase_new": mase_windows(pn, ws, C, H), "mase_old": mase_windows(po, ws, C, H)})
    d.to_csv(f"results/timesfm_uni_fix/{name}_uni.csv", index=False)
    print(name, len(d), "GM old", round(float(np.exp(np.log(d.mase_old).mean())), 4), "new", round(float(np.exp(np.log(d.mase_new).mean())), 4), f"{(time.time()-T0)/60:.1f} min", flush=True)
print("DONE")
