# ---- Chronos-2 fine-tuning baselines on RiverLead (raw past covariate). CFG + lta/ injected above. ----
import time, subprocess, traceback, glob
T0 = time.time()
OUT = _W
def P(*a):
    print(f"PROGRESS [{(time.time()-T0)/60:6.1f} min]", *a, flush=True)
if CFG.get("local") != "1":
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chronos-forecasting==2.3.2", "peft"],
                       capture_output=True, text=True); P("pip rc", r.returncode)
    r = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], capture_output=True, text=True)
    P("uninstall torchao rc", r.returncode)
import numpy as np, pandas as pd, torch
from lta import backbones as B
from lta import river as RV
from lta.arms import arm_covariates, mase_windows

DEV = CFG.get("device", "cuda"); C, H, K = 512, 24, 168
def find(name):
    return (glob.glob(f"/kaggle/input/**/{name}", recursive=True) + glob.glob(f"data/river/{name}"))[0]
Hd = RV.load_hourly(find("hourly.csv")); pairs = json.load(open(find("pairs.json")))
lags_cls = json.load(open(find("classical_lags_train.json")))
W = {s: RV.make_windows(Hd, pairs, s, C=C, H=H) for s in ["test"]}
if CFG.get("river_sub"):
    W = {k: v[::max(1, len(v) // int(CFG["river_sub"]))] for k, v in W.items()}
gm = lambda v: float(np.exp(np.mean(np.log(np.asarray(v)))))

def segments(split, min_len=C + H + 48):
    a, b = RV.SPLITS[split]; out = []
    for name, (up, dn) in pairs.items():
        d = Hd[[up, dn]].interpolate(limit=6, limit_area="inside")
        d = d[(d.index >= pd.Timestamp(a, tz=d.index.tz)) & (d.index < pd.Timestamp(b, tz=d.index.tz))]
        ok = d.notna().all(1).values
        edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            if e - s >= min_len:
                out.append({"target": d[dn].values[s:e].astype(np.float32),
                            "past_covariates": {"up": d[up].values[s:e].astype(np.float32)}})
    return out

tr, va = segments("train"), segments("val")
P("segments", len(tr), len(va), sum(len(s["target"]) for s in tr))
SUMMARY = {"cfg": CFG, "runs": {}}
rows = []
base = B.Chronos2(CFG.get("chronos_id", "amazon/chronos-2"), device=DEV)
MODES = CFG.get("modes", "full,lora").split(",")
for mode, lr in [(m_, float(CFG.get("lr_" + m_, 1e-6 if m_ == "full" else 1e-5))) for m_ in MODES]:
    try:
        t1 = time.time()
        newpipe = base.pipe.fit(inputs=tr, prediction_length=H, validation_inputs=va, finetune_mode=mode,
                                context_length=C, learning_rate=lr, num_steps=int(CFG.get("steps", 1000)),
                                batch_size=int(CFG.get("ft_bs", 256)), output_dir=os.path.join(OUT, f"ft_{mode}"),
                                report_to="none")
        P("fit done", mode, f"{(time.time()-t1)/60:.1f} min")
        wrap = B.Chronos2.__new__(B.Chronos2)
        wrap.__dict__.update(base.__dict__); wrap.pipe = newpipe; wrap.m = newpipe.model.eval()
        for arm in ["uni", "raw", "binned"]:
            y, cc, cf, _ = arm_covariates(W["test"], arm, C, H, K, global_lags=lags_cls["global"], binned=lags_cls["binned"])
            pred = wrap.forecast(y, cc, cf, H=H)
            ms = mase_windows(pred, W["test"], C, H)
            for i, w in enumerate(W["test"]):
                rows.append({"set": "river_test", "arm": f"ft_{mode}_{arm}", "cell": w["cell"], "id": w["meta"]["origin"],
                             "event": w["meta"]["event"], "mase": float(ms[i])})
            P(mode, arm, "GM-MASE", round(gm(ms), 4))
        SUMMARY["runs"][mode] = {"lr": lr, "min": (time.time() - t1) / 60}
        del newpipe, wrap
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        import shutil; shutil.rmtree(os.path.join(OUT, f"ft_{mode}"), ignore_errors=True)
    except Exception as e:
        SUMMARY["runs"][mode] = {"err": repr(e), "tb": traceback.format_exc()[-3000:]}; P("FAIL", mode, repr(e)[:300])
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "ft_rows.csv"), index=False)
    json.dump(SUMMARY, open(os.path.join(OUT, "ft_summary.json"), "w"), indent=1, default=str)
P("DONE")
