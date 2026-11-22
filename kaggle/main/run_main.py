# ---- Main LTA kernel: synthetic training, synthetic test, RiverLead (zs / ft / lopo). CFG + lta/ injected. ----
import time, subprocess, traceback, glob
T0 = time.time()
OUT = _W
def P(*a):
    print(f"PROGRESS [{(time.time()-T0)/60:6.1f} min]", *a, flush=True)
def pip(*args):
    if CFG.get("local") == "1": return 0
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], capture_output=True, text=True)
    P("pip", args[:2], "rc", r.returncode); return r.returncode
pip("chronos-forecasting==2.3.2")
import numpy as np, pandas as pd, torch
from lta.synth import make_task, GATE_CELLS
from lta import backbones as B
from lta.train import build_pool, train_lta, batch_arrays, lta_covariates
from lta.arms import arm_covariates, mase_windows
from lta import river as RV

DEV = CFG.get("device", "cuda")
C, H, K = 512, 24, 168
S = {k: int(CFG.get(k, v)) for k, v in dict(pool_n=30000, synth_steps=3000, ft_steps=1500, lopo_steps=600,
                                             n_test_cell=100, bs=32, eval_every=250).items()}
STAGES = CFG.get("stages", "synth,synth_test,river,lopo").split(",")
SEED = int(CFG.get("seed", 0)); DYN = CFG.get("dynamic", "1") == "1"
AUX = float(CFG.get("aux", 1.0)); CCF0 = float(CFG.get("ccf_init", 10.0)); TAG = CFG.get("tag", "")
SUMMARY = {"cfg": CFG, "stages": {}}
def dump():
    SUMMARY["elapsed_min"] = (time.time() - T0) / 60
    json.dump(SUMMARY, open(os.path.join(OUT, "main_summary.json"), "w"), indent=1, default=str)

bb = B.Chronos2(CFG.get("chronos_id", "amazon/chronos-2"), device=DEV)
gm = lambda v: float(np.exp(np.mean(np.log(np.asarray(v)))))

@torch.no_grad()
def run_bb(y, cc, cf, bs=128):
    """Chronos-2 -> (median (N,H), quantiles (N,Q,H))."""
    qs = []
    for s in range(0, len(y), bs):
        yt = torch.tensor(y[s:s + bs], device=DEV)
        ct = None if cc is None else torch.tensor(cc[s:s + bs], device=DEV)
        ft = None if cf is None else torch.tensor(cf[s:s + bs], device=DEV)
        qs.append(bb.forward_tensors(yt, ct, ft, H).float().cpu().numpy())
    q = np.concatenate(qs, 0)
    return q[:, bb.qi], q

def wql(q, windows):
    lv = np.array(bb.Q)[None, :, None]
    yf = np.stack([w["y"][C:C + H] for w in windows])[:, None, :]
    d = yf - q
    pin = np.maximum(lv * d, (lv - 1) * d)
    return 2 * pin.sum(-1).mean(-1) / (np.abs(yf[:, 0]).sum(-1) + 1e-8)

@torch.no_grad()
def lta_align(model, windows, bs=64):
    out = []
    for s in range(0, len(windows), bs):
        y, x = batch_arrays(windows[s:s + bs], C, H)
        out.append(model(torch.tensor(y[:, :C], device=DEV), torch.tensor(x, device=DEV), H).float().cpu().numpy())
    return np.concatenate(out, 0)

def eval_arms(windows, arms, tag, extra=None, global_lags=None, binned=None):
    rows = []
    for arm in arms:
        t1 = time.time()
        if arm in (extra or {}):
            y, cc, cf, lags = arm_covariates(windows, "lta", C, H, lta_aligned=extra[arm])
        else:
            y, cc, cf, lags = arm_covariates(windows, arm, C, H, K, global_lags=global_lags, binned=binned)
        med, q = run_bb(y, cc, cf)
        ms = mase_windows(med, windows, C, H); wq = wql(q, windows)
        for i, w in enumerate(windows):
            rows.append({"set": tag, "arm": arm, "cell": w["cell"], "id": w["meta"].get("origin", w["meta"].get("index")),
                         "event": w["meta"].get("event", None), "mase": float(ms[i]), "wql": float(wq[i]),
                         "lag": (lags[i][0] if lags is not None and len(lags[i]) else -1)})
        P(tag, arm, "GM-MASE", round(gm(ms), 4), f"{time.time()-t1:.0f}s")
    return rows

def save_rows(rows, name):
    pd.DataFrame(rows).to_csv(os.path.join(OUT, name), index=False)

def make_val_fn(windows):
    def f(m):
        al = lta_align(m, windows)
        y, cc, cf, _ = arm_covariates(windows, "lta", C, H, lta_aligned=al)
        med, _ = run_bb(y, cc, cf)
        return gm(mase_windows(med, windows, C, H))
    return f

# ------------------------------------------------------------------ stage 1: synthetic training
lta_zs = None
if "synth" in STAGES:
    try:
        t1 = time.time()
        pool = build_pool(S["pool_n"], "train", C, H, seed=0)
        valset = build_pool(400, "val", C, H, seed=1)
        for w in pool + valset: w["cell"] = w["meta"]["family"]
        P("pool built", len(pool), f"{time.time()-t1:.0f}s")
        lta_zs, hist, bstep = train_lta(bb, pool, make_val_fn(valset), C=C, H=H, K=K, steps=S["synth_steps"],
                                        bs=S["bs"], lr=2e-3, eval_every=S["eval_every"], log=P, device=DEV, seed=SEED,
                                        time_budget_min=float(CFG.get("synth_budget_min", 150)), dynamic=DYN,
                                        aux_ce=AUX, aux_mse=AUX, ccf_init_scale=CCF0)
        torch.save(lta_zs.state_dict(), os.path.join(OUT, "lta_zs.pt"))
        SUMMARY["stages"]["synth"] = {"best_step": bstep, "hist": hist}
    except Exception as e:
        SUMMARY["stages"]["synth"] = {"err": repr(e), "tb": traceback.format_exc()[-3000:]}; P("FAIL synth", repr(e))
    dump()
elif os.path.exists(CFG.get("lta_zs_path", "")):
    from lta.aligner import LeadTimeAligner
    lta_zs = LeadTimeAligner(K=K, dynamic=DYN, ccf_init_scale=CCF0).to(DEV); lta_zs.load_state_dict(torch.load(CFG["lta_zs_path"], map_location=DEV))

# ------------------------------------------------------------------ stage 2: synthetic test
if "synth_test" in STAGES and lta_zs is not None:
    try:
        lta_zs.eval()
        cells = GATE_CELLS + [("null", 24, None), ("linear", 12, None), ("tvlag", 72, 30)]
        test = []
        for fam, tau0, d in cells:
            for i in range(S["n_test_cell"]):
                T = make_task(fam, tau0, "test", i, C=C, H=H, delta=d)
                T["cell"] = f"{fam}_{tau0}" + (f"_d{d}" if d else ""); test.append(T)
        oracle = []
        for T in test:
            xo = T["xo"].copy()
            src = np.arange(C + H) - T["tau"]
            xo[(src < 0) | (src >= C)] = np.nan              # only observed sources are usable
            oracle.append(xo[None])
        extra = {"oracle": np.stack(oracle), "lta_zs": lta_align(lta_zs, test)}
        arms_st = ["uni", "raw", "ccf", "ccf_pw", "gcc_phat", "top3", "randlag", "oracle", "lta_zs"]
        if CFG.get("skip_classical") == "1": arms_st = ["lta_zs"]
        rows = eval_arms(test, arms_st, "synth_test", extra)
        save_rows(rows, "synth_test_rows.csv")
        SUMMARY["stages"]["synth_test"] = {"n": len(test)}
    except Exception as e:
        SUMMARY["stages"]["synth_test"] = {"err": repr(e), "tb": traceback.format_exc()[-3000:]}; P("FAIL synth_test", repr(e))
    dump()

# ------------------------------------------------------------------ stage 3: rivers
def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) + glob.glob(f"data/river/{name}")
    return hits[0]
if "river" in STAGES or "lopo" in STAGES:
    Hd = RV.load_hourly(find("hourly.csv")); pairs = json.load(open(find("pairs.json")))
    lags_cls = json.load(open(find("classical_lags_train.json")))
    W = {s: RV.make_windows(Hd, pairs, s, C=C, H=H) for s in ["train", "val", "test"]}
    if CFG.get("river_sub"):                                   # local dry runs only
        n_ = int(CFG["river_sub"]); W = {k: v[::max(1, len(v) // n_)] for k, v in W.items()}
    rng = np.random.default_rng(0)
    val_sub = [W["val"][i] for i in sorted(rng.choice(len(W["val"]), min(600, len(W["val"])), replace=False))]
    P("river windows", {k: len(v) for k, v in W.items()})
    CL_ARMS = ["uni", "raw", "ccf", "ccf_pw", "gcc_phat", "top3", "global", "binned", "randlag"]

if "river" in STAGES:
    try:
        rows = []
        for split in (["val", "test"] if CFG.get("skip_classical") != "1" else []):
            rows += eval_arms(W[split], CL_ARMS, "river_" + split, global_lags=lags_cls["global"], binned=lags_cls["binned"])
            save_rows(rows, "river_rows.csv")
        if lta_zs is not None:
            lta_zs.eval()
            for split in ["val", "test"]:
                rows += eval_arms(W[split], ["lta_zs"], "river_" + split, {"lta_zs": lta_align(lta_zs, W[split])})
            save_rows(rows, "river_rows.csv")
            lta_ft, hist, bstep = train_lta(bb, W["train"], make_val_fn(val_sub), init_state=lta_zs.state_dict(), C=C, H=H,
                                            K=K, steps=S["ft_steps"], bs=S["bs"], lr=1e-3, eval_every=150, log=P,
                                            device=DEV, seed=SEED + 1, aux_ce=0, aux_mse=0, dynamic=DYN, ccf_init_scale=CCF0)
            torch.save(lta_ft.state_dict(), os.path.join(OUT, "lta_ft.pt"))
            SUMMARY["stages"]["river_ft"] = {"best_step": bstep, "hist": hist}
            lta_ft.eval()
            al = {}
            for split in ["val", "test"]:
                al[split] = lta_align(lta_ft, W[split])
                rows += eval_arms(W[split], ["lta_ft"], "river_" + split, {"lta_ft": al[split]})
            save_rows(rows, "river_rows.csv")
            np.savez_compressed(os.path.join(OUT, "aligned_river_test.npz"),
                                lta_ft=al["test"].astype(np.float16), lta_zs=lta_align(lta_zs, W["test"]).astype(np.float16))
        SUMMARY["stages"]["river"] = {"n": {k: len(v) for k, v in W.items()}}
    except Exception as e:
        SUMMARY["stages"]["river"] = {"err": repr(e), "tb": traceback.format_exc()[-3000:]}; P("FAIL river", repr(e))
    dump()

# ------------------------------------------------------------------ stage 4: leave-one-pair-out
if "lopo" in STAGES and lta_zs is not None:
    rows, al_lopo = [], np.zeros((len(W["test"]), 1, C + H), np.float16)
    for p in sorted(pairs):
        try:
            tr = [w for w in W["train"] if w["cell"] != p]
            vs = [w for w in val_sub if w["cell"] != p]
            te_idx = [i for i, w in enumerate(W["test"]) if w["cell"] == p]
            te = [W["test"][i] for i in te_idx]
            m, hist, bstep = train_lta(bb, tr, make_val_fn(vs), init_state=lta_zs.state_dict(), C=C, H=H, K=K,
                                       steps=S["lopo_steps"], bs=S["bs"], lr=1e-3, eval_every=150, log=P, device=DEV,
                                       seed=SEED + 2, aux_ce=0, aux_mse=0, dynamic=DYN, ccf_init_scale=CCF0)
            m.eval(); a = lta_align(m, te); al_lopo[te_idx] = a.astype(np.float16)
            rows += eval_arms(te, ["lta_lopo"], "river_test", {"lta_lopo": a})
            SUMMARY["stages"].setdefault("lopo", {})[p] = {"best_step": bstep}
            save_rows(rows, "river_lopo_rows.csv")
        except Exception as e:
            SUMMARY["stages"].setdefault("lopo", {})[p] = {"err": repr(e)}; P("FAIL lopo", p, repr(e))
        dump()
    np.savez_compressed(os.path.join(OUT, "aligned_river_test_lopo.npz"), lta_lopo=al_lopo)
P("DONE"); dump()
