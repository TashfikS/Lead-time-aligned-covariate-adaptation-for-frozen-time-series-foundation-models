"""CPU replacement for the queued Kaggle transfer kernels (Moirai-1.1 base; Chronos-2 on float32 aligned arrays).
Same windows, arms and arm construction as kaggle/transfer/run_transfer.py; only the device differs (D8)."""
import sys, os, json, time
sys.path.insert(0, ".")
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, pandas as pd, torch
from lta import backbones as B, river as RV
from lta.arms import arm_covariates, mase_windows
torch.set_num_threads(4)
C, H, K = 512, 24, 168
which = sys.argv[1]
Hd = RV.load_hourly("data/river/hourly.csv"); pairs = json.load(open("data/river/pairs.json"))
lags = json.load(open("data/river/classical_lags_train.json"))
test = RV.make_windows(Hd, pairs, "test", C=C, H=H)
al = np.load("build/ds_aligned/aligned_river_test.npz")["lta_zs"].astype(np.float32)
assert al.shape[0] == len(test), (al.shape, len(test))
if which == "moirai11":
    from uni2ts.model.moirai import MoiraiModule, MoiraiForecast
    m = B.Moirai11.__new__(B.Moirai11); m.torch = torch; m.MF = MoiraiForecast
    m.mod = MoiraiModule.from_pretrained("models/moirai-1.1-R-base"); m.device = "cpu"; m.ns = 100
    arms = ["uni", "raw", "top3", "binned", "lta_zs"]
else:
    m = B.Chronos2("models/chronos-2", device="cpu"); arms = ["lta_zs"]
out = f"results/tr_{which}_local"; os.makedirs(out, exist_ok=True)
rows = []; T0 = time.time()
for arm in arms:
    if arm == "lta_zs":
        y, cc, cf, _ = arm_covariates(test, "lta", C, H, lta_aligned=al)
    else:
        y, cc, cf, _ = arm_covariates(test, arm, C, H, K, global_lags=lags["global"], binned=lags["binned"])
    preds = []
    for s in range(0, len(test), 256):
        preds.append(m.forecast(y[s:s+256], None if cc is None else cc[s:s+256], None if cf is None else cf[s:s+256], H=H))
    ms = mase_windows(np.concatenate(preds, 0), test, C, H)
    for i, w in enumerate(test):
        rows.append({"set": "river_test", "backbone": which, "arm": arm, "cell": w["cell"], "id": w["meta"]["origin"],
                     "event": w["meta"]["event"], "mase": float(ms[i])})
    pd.DataFrame(rows).to_csv(f"{out}/transfer_{which}_rows.csv", index=False)
    print(which, arm, "GM", round(float(np.exp(np.log(ms).mean())), 4), f"{(time.time()-T0)/60:.1f} min", flush=True)
print("DONE", flush=True)
