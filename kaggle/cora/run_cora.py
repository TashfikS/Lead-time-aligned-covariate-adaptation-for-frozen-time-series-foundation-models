# ---- P5-D: CoRA frozen adapter on Chronos-2 (RiverLead, AirLead, EnergyLead). CFG + lta/ injected. ----
import time, subprocess, traceback, glob
T0 = time.time(); OUT = _W
def P(*a): print(f"PROGRESS [{(time.time()-T0)/60:6.1f} min]", *a, flush=True)
if CFG.get("local") != "1":
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chronos-forecasting==2.3.2"], capture_output=True, text=True); P("pip", r.returncode)
import numpy as np, pandas as pd, torch
from lta import backbones as B, river as RV, cora as CR
from lta.arms import mase_windows
DEV = CFG.get("device", "cuda"); C, H = 512, 24
STEPS, BS, EV = int(CFG.get("steps", 2000)), int(CFG.get("bs", 64)), int(CFG.get("eval_every", 250))
def find(ds, fn):
    if CFG.get("local") == "1":
        return os.path.join(CFG["root"], {"lta-riverlead": "data/river", "lta-airlead": "data/weather", "lta-energylead": "data/energy", "lta-trafficlead": "data/traffic"}[ds], fn)
    c = [p for p in glob.glob(f"/kaggle/input/**/{fn}", recursive=True) if ds in p]
    assert c, (ds, fn); return c[0]
DOM = {"river": ("lta-riverlead", {}), "air": ("lta-airlead", {"event_kind": "delta", "event_delta": 8.0}),
       "energy": ("lta-energylead", {"event_kind": "relrange", "splits": RV.ENERGY_SPLITS}),
       "traffic": ("lta-trafficlead", dict(event_kind="delta", event_delta=20.0, splits=RV.TRAFFIC_SPLITS, step_minutes=5))}
bb = B.Chronos2(CFG.get("chronos_id", "amazon/chronos-2"), device=DEV); m = bb.m
gm = lambda v: float(np.exp(np.mean(np.log(np.asarray(v)))))
rows, summ = [], {"cfg": CFG}
for dom in CFG.get("domains", "river,air,energy").split(","):
    try:
        dsn, kw = DOM[dom]
        Hd = RV.load_hourly(find(dsn, "hourly.csv")); pairs = json.load(open(find(dsn, "pairs.json")))
        W = {s: RV.make_windows(Hd, pairs, s, C=C, H=H, **kw) for s in ["train", "val", "test"]}
        if CFG.get("sub"): W = {k: v[::max(1, len(v) // int(CFG["sub"]))] for k, v in W.items()}
        P(dom, {k: len(v) for k, v in W.items()})
        EM = {}
        for s, ws in W.items():
            Y = torch.tensor(np.stack([w["y"][:C] for w in ws]), device=DEV); X = torch.tensor(np.stack([w["x"][:C] for w in ws]), device=DEV)
            Et, Ec, loc, sc = CR.embed(m, Y, X, H)
            Yf = torch.tensor(np.stack([w["y"][C:] for w in ws]), device=DEV)
            yn = m.instance_norm(Yf, (loc, sc))[0]
            EM[s] = (Et, Ec, loc, sc, yn)
        P(dom, "embedded")
        def predict(head, s, zero_cov):
            Et, Ec, loc, sc, _ = EM[s]; out = []
            with torch.no_grad():
                for a in range(0, len(Et), 512):
                    ec = torch.zeros_like(Ec[a:a+512]) if zero_cov else Ec[a:a+512]
                    out.append(head.unscale(head(Et[a:a+512], ec), loc[a:a+512], sc[a:a+512])[:, bb.qi].float().cpu().numpy())
            return np.concatenate(out)
        for arm, zero_cov in [("cora_raw", False), ("cora_nocov", True)]:
            torch.manual_seed(0); head = CR.CoRAHead(m, H).to(DEV)
            params = [p for n, p in head.named_parameters() if not n.startswith("m.")]
            opt = torch.optim.AdamW(params, lr=1e-3)
            v0 = gm(mase_windows(predict(head, "val", zero_cov), W["val"], C, H)); best = (v0, 0, {k: v.detach().clone() for k, v in head.state_dict().items() if not k.startswith("m.")})
            P(dom, arm, "step 0 val", round(v0, 4)); hist = [(0, v0)]
            Et, Ec, loc, sc, yn = EM["train"]; g = torch.Generator(device="cpu").manual_seed(0)
            for step in range(1, STEPS + 1):
                ix = torch.randint(0, len(Et), (BS,), generator=g).to(DEV)
                ec = torch.zeros_like(Ec[ix]) if zero_cov else Ec[ix]
                loss = CR.pinball(head(Et[ix], ec), yn[ix], m.quantiles.to(DEV))
                opt.zero_grad(); loss.backward(); opt.step()
                if step % EV == 0:
                    v = gm(mase_windows(predict(head, "val", zero_cov), W["val"], C, H)); hist.append((step, v))
                    if v < best[0]: best = (v, step, {k: vv.detach().clone() for k, vv in head.state_dict().items() if not k.startswith("m.")})
                    P(dom, arm, "step", step, "loss", round(float(loss), 4), "val", round(v, 4), "best", round(best[0], 4), "@", best[1])
            head.load_state_dict(best[2], strict=False)
            ms = mase_windows(predict(head, "test", zero_cov), W["test"], C, H)
            for i, w in enumerate(W["test"]):
                rows.append({"set": f"{dom}_test", "arm": arm, "cell": w["cell"], "id": w["meta"]["origin"], "event": w["meta"]["event"], "mase": float(ms[i])})
            summ[f"{dom}/{arm}"] = {"best_step": best[1], "best_val": best[0], "val0": v0, "hist": hist, "test_gm": gm(ms)}
            P(dom, arm, "TEST GM-MASE", round(gm(ms), 4), "best step", best[1])
        # sanity: uni at init equals cora step 0 (already val0); record test uni via frozen forecast for a subset
        del EM; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        summ[dom] = {"err": repr(e), "tb": traceback.format_exc()[-3000:]}; P("FAIL", dom, repr(e)[:300])
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cora_rows.csv"), index=False)
    json.dump(summ, open(os.path.join(OUT, "cora_summary.json"), "w"), indent=1, default=str)
P("DONE")
