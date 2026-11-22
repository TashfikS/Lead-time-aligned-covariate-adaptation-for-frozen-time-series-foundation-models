# ---- AirLead (P4): frozen synthetic-only LTA aligners + classical arms on a frozen backbone. CFG + lta/ injected. ----
import time, subprocess, traceback, glob
T0 = time.time()
OUT = _W
def P(*a):
    print(f"PROGRESS [{(time.time()-T0)/60:6.1f} min]", *a, flush=True)
def pip(*args):
    if CFG.get("local") == "1": return 0
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], capture_output=True, text=True)
    P("pip", args[:2], "rc", r.returncode); return r.returncode
BB = CFG["backbone"]
pip("chronos-forecasting==2.3.2")
if BB == "timesfm25": pip("timesfm[torch,xreg]")
import numpy as np, pandas as pd, torch
from lta import backbones as B
from lta import river as RV
from lta.arms import arm_covariates, mase_windows, fit_global_lag, fit_binned_lag
from lta.aligner import LeadTimeAligner
from lta.train import lta_aligned_numpy
C, H, K = 512, 24, 168
DEV = CFG.get("device", "cuda")
def find(name):
    return (glob.glob(f"/kaggle/input/**/{name}", recursive=True) + glob.glob(f"data/weather/{name}") + glob.glob(name))[0]
Hd = RV.load_hourly(find("hourly.csv")); pairs = json.load(open(find("pairs.json")))
DOMAIN = CFG.get("domain", "air")
WKW = {"air": dict(event_kind="delta", event_delta=8.0), "energy": dict(event_kind="relrange", splits=RV.ENERGY_SPLITS),
       "traffic": dict(event_kind="delta", event_delta=20.0, splits=RV.TRAFFIC_SPLITS, step_minutes=5)}[DOMAIN]
W = {s: RV.make_windows(Hd, pairs, s, C=C, H=H, **WKW) for s in ["train", "val", "test"]}
if CFG.get("sub"):
    W = {k: v[::max(1, len(v) // int(CFG["sub"]))] for k, v in W.items()}
P("windows", {k: len(v) for k, v in W.items()}, "event frac test", np.mean([w["meta"]["event"] for w in W["test"]]).round(3))
GL = fit_global_lag(W["train"], C); BN = fit_binned_lag(W["train"], C)
json.dump({"global": GL, "binned": BN}, open(os.path.join(OUT, f"{DOMAIN}_classical_lags.json"), "w"), indent=1, default=str)
P("global lags", GL)
ALIGNED = {}
for tag in CFG.get("seeds", "s0,s1,s2").split(","):
    m = LeadTimeAligner(K=K, dynamic=True, ccf_init_scale=10.0).to(DEV)
    m.load_state_dict(torch.load(find(f"lta_zs_{tag}.pt"), map_location=DEV)); m.eval()
    for split in CFG.get("splits", "val,test").split(","):
        ALIGNED[(f"lta_{tag}", split)] = lta_aligned_numpy(m, W[split], C, H, bs=64, device=DEV).astype(np.float32)
    P("aligned", tag)
    del m
model = {"chronos2": lambda: B.Chronos2(CFG.get("chronos_id", "amazon/chronos-2"), device=DEV),
         "timesfm25": lambda: B.TimesFM25()}[BB]()
gm = lambda v: float(np.exp(np.mean(np.log(np.asarray(v)))))
rows, summ = [], {"cfg": CFG, "res": {}}
ARMS = CFG["arms"].split(",")
for split in CFG.get("splits", "val,test").split(","):
    ws = W[split]
    for arm in ARMS:
        t1 = time.time()
        try:
            if arm.startswith("lta"):
                y, cc, cf, _ = arm_covariates(ws, "lta", C, H, lta_aligned=ALIGNED[(arm, split)])
            else:
                y, cc, cf, _ = arm_covariates(ws, arm, C, H, K, global_lags=GL, binned=BN)
            preds = []
            for s in range(0, len(ws), 1000):
                preds.append(model.forecast(y[s:s + 1000], None if cc is None else cc[s:s + 1000],
                                            None if cf is None else cf[s:s + 1000], H=H))
            ms = mase_windows(np.concatenate(preds, 0), ws, C, H)
            for i, w in enumerate(ws):
                rows.append({"set": DOMAIN + "_" + split, "backbone": BB, "arm": arm, "cell": w["cell"], "id": w["meta"]["origin"],
                             "event": w["meta"]["event"], "mase": float(ms[i])})
            summ["res"][f"{split}/{arm}"] = gm(ms)
            P(BB, split, arm, "GM-MASE", round(gm(ms), 4), f"{time.time()-t1:.0f}s")
        except Exception as e:
            summ["res"][f"{split}/{arm}"] = {"err": repr(e), "tb": traceback.format_exc()[-2000:]}; P("FAIL", split, arm, repr(e)[:300])
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{DOMAIN}_{BB}_rows.csv"), index=False)
        json.dump(summ, open(os.path.join(OUT, f"{DOMAIN}_{BB}_summary.json"), "w"), indent=1, default=str)
P("DONE")
