# ---- Transfer: aligned covariates from the frozen LTA (trained through Chronos-2) fed to other frozen backbones.
import time, subprocess, traceback, glob
T0 = time.time()
OUT = _W
def P(*a):
    print(f"PROGRESS [{(time.time()-T0)/60:6.1f} min]", *a, flush=True)
def pip(*args):
    if CFG.get("local") == "1": return 0
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], capture_output=True, text=True)
    P("pip", args[:3], "rc", r.returncode, r.stderr[-200:].replace("\n", " ")); return r.returncode
BB = CFG["backbone"]
if BB == "timesfm25": pip("timesfm[torch,xreg]")
if BB == "tabpfn_ts": pip("tabpfn-time-series")
if BB == "moirai11":
    pip("--no-deps", "uni2ts"); pip("gluonts", "jaxtyping", "einops", "hydra-core==1.3.2", "omegaconf==2.3.0", "lightning", "safetensors")
if BB == "chronos2": pip("chronos-forecasting==2.3.2")
import numpy as np, pandas as pd
from lta import backbones as B
from lta import river as RV
from lta.arms import arm_covariates, mase_windows
C, H, K = 512, 24, 168
def find(name):
    return (glob.glob(f"/kaggle/input/**/{name}", recursive=True) + glob.glob(f"data/river/{name}") + glob.glob(name))[0]
Hd = RV.load_hourly(find("hourly.csv")); pairs = json.load(open(find("pairs.json")))
lags_cls = json.load(open(find("classical_lags_train.json")))
test = RV.make_windows(Hd, pairs, "test", C=C, H=H)
al = dict(np.load(find("aligned_river_test.npz")))
try:
    al.update(dict(np.load(find("aligned_river_test_lopo.npz"))))
except Exception:
    pass
EVERY = int(CFG.get("every", 1))
idx = [i for i in range(len(test)) if i % EVERY == 0]      # deterministic subsample (same for all arms)
PART, NPARTS = int(CFG.get("part", 0)), int(CFG.get("nparts", 1))
idx = idx[PART::NPARTS]                                      # disjoint deterministic parts; union = full subsample
sub = [test[i] for i in idx]
P("backbone", BB, "windows", len(sub), "aligned arrays", list(al))
gm = lambda v: float(np.exp(np.mean(np.log(np.asarray(v)))))
model = {"timesfm25": lambda: B.TimesFM25(), "tabpfn_ts": lambda: B.TabPFNTS("v2"),
         "moirai11": lambda: B.Moirai11("base", device="cuda"),
         "chronos2": lambda: B.Chronos2("amazon/chronos-2", device="cuda")}[BB]()
ARMS = CFG.get("arms", "uni,raw,ccf_pw,binned,lta_zs,lta_ft,lta_lopo").split(",")
rows, summ = [], {"cfg": CFG, "arms": {}}
CH = int(CFG.get("chunk", 500))
for arm in ARMS:
    t1 = time.time()
    try:
        if arm.startswith("lta"):
            if arm not in al: P("skip", arm); continue
            y, cc, cf, _ = arm_covariates(sub, "lta", C, H, lta_aligned=al[arm][idx].astype(np.float32))
        else:
            y, cc, cf, _ = arm_covariates(sub, arm, C, H, K, global_lags=lags_cls["global"], binned=lags_cls["binned"])
        preds = []
        for s in range(0, len(sub), CH):
            e = s + CH
            preds.append(model.forecast(y[s:e], None if cc is None else cc[s:e], None if cf is None else cf[s:e], H=H))
            P(arm, e, "/", len(sub))
        pred = np.concatenate(preds, 0)
        ms = mase_windows(pred, sub, C, H)
        for i, w in enumerate(sub):
            rows.append({"set": "river_test", "backbone": BB, "arm": arm, "cell": w["cell"], "id": w["meta"]["origin"],
                         "event": w["meta"]["event"], "mase": float(ms[i])})
        summ["arms"][arm] = {"gm": gm(ms), "min": (time.time() - t1) / 60}
        P(BB, arm, "GM-MASE", round(gm(ms), 4))
    except Exception as e:
        summ["arms"][arm] = {"err": repr(e), "tb": traceback.format_exc()[-2500:]}; P("FAIL", arm, repr(e)[:300])
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"transfer_{BB}_rows.csv"), index=False)
    json.dump(summ, open(os.path.join(OUT, f"transfer_{BB}_summary.json"), "w"), indent=1, default=str)
P("DONE")
