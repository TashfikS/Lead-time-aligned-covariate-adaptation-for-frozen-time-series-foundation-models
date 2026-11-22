# ---- P5-A: LeadBench-S test on another frozen backbone with the frozen synthetic-only aligners. CFG + lta/ injected. ----
import time, subprocess, traceback, glob
T0 = time.time(); OUT = _W
def P(*a): print(f"PROGRESS [{(time.time()-T0)/60:6.1f} min]", *a, flush=True)
BB = CFG["backbone"]
def pip(*a):
    if CFG.get("local") == "1": return
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a], capture_output=True, text=True); P("pip", a[:1], r.returncode)
if BB == "timesfm25": pip("timesfm[torch,xreg]")
if BB == "tabpfn_ts": pip("tabpfn-time-series")
import numpy as np, pandas as pd, torch
from lta.synth import make_task, GATE_CELLS
from lta import backbones as B
from lta.arms import arm_covariates, mase_windows
C, H, K = 512, 24, 168
NPC = int(CFG.get("n_per_cell", 100))
cells = GATE_CELLS + [("null", 24, None), ("linear", 12, None), ("tvlag", 72, 30)]
test = []
for fam, tau0, d in cells:
    for i in range(100):
        T = make_task(fam, tau0, "test", i, C=C, H=H, delta=d); T["cell"] = f"{fam}_{tau0}" + (f"_d{d}" if d else "")
        if i < NPC: test.append(T)
sel = np.array([ci * 100 + i for ci in range(len(cells)) for i in range(NPC)])
alp = [p for p in glob.glob("/kaggle/input/**/aligned_synth_test.npz", recursive=True)] or ["aligned_synth_test.npz"]
AL = {k: v[sel].astype(np.float32) for k, v in dict(np.load(alp[0])).items()}
oracle = []
for T in test:
    xo = T["xo"].copy(); src = np.arange(C + H) - T["tau"]; xo[(src < 0) | (src >= C)] = np.nan; oracle.append(xo[None])
AL["oracle"] = np.stack(oracle).astype(np.float32)
P("backbone", BB, "series", len(test), "aligned", list(AL))
model = {"timesfm25": lambda: B.TimesFM25(), "tabpfn_ts": lambda: B.TabPFNTS("v2")}[BB]()
ARMS = CFG["arms"].split(","); rows, summ = [], {"cfg": CFG, "arms": {}}
gm = lambda v: float(np.exp(np.mean(np.log(np.asarray(v)))))
CH = int(CFG.get("chunk", 260))
for arm in ARMS:
    t1 = time.time()
    try:
        if arm in AL: y, cc, cf, _ = arm_covariates(test, "lta", C, H, lta_aligned=AL[arm])
        else: y, cc, cf, _ = arm_covariates(test, arm, C, H, K)
        preds = []
        for s in range(0, len(test), CH):
            preds.append(model.forecast(y[s:s+CH], None if cc is None else cc[s:s+CH], None if cf is None else cf[s:s+CH], H=H))
            P(arm, s + CH, "/", len(test))
        pred = np.concatenate(preds); ms = mase_windows(pred, test, C, H)
        for i, w in enumerate(test):
            rows.append({"set": "synth_test", "backbone": BB, "arm": arm, "cell": w["cell"], "id": w["meta"]["index"], "mase": float(ms[i])})
        summ["arms"][arm] = {"gm": gm(ms), "s": time.time() - t1}; P(BB, arm, "GM-MASE", round(gm(ms), 4))
    except Exception as e:
        summ["arms"][arm] = {"err": repr(e), "tb": traceback.format_exc()[-2500:]}; P("FAIL", arm, repr(e)[:300])
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"synthtr_{BB}_rows.csv"), index=False)
    json.dump(summ, open(os.path.join(OUT, f"synthtr_{BB}_summary.json"), "w"), indent=1, default=str)
P("DONE")
