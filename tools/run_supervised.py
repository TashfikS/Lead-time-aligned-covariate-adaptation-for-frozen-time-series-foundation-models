"""P5-C: ARX / LightGBM per pair on RiverLead, AirLead, EnergyLead (full and 90-day history; with / without covariate)."""
import sys, os, json, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from lta import river as RV, supervised as SV
from lta.arms import mase_windows
DS = {"river": ("data/river/hourly.csv", "data/river/pairs.json", RV.SPLITS, {}),
      "air": ("data/weather/hourly.csv", "data/weather/pairs.json", RV.SPLITS, {"event_kind": "delta", "event_delta": 8.0}),
      "energy": ("data/energy/hourly.csv", "data/energy/pairs.json", RV.ENERGY_SPLITS, {"event_kind": "relrange", "splits": RV.ENERGY_SPLITS}),
      "traffic": ("data/traffic/hourly.csv", "data/traffic/pairs.json", RV.TRAFFIC_SPLITS, dict(event_kind="delta", event_delta=20.0, splits=RV.TRAFFIC_SPLITS, step_minutes=5))}
SHORT_DAYS = {"traffic": 14}                      # P6: train split is only 90 days long
os.makedirs("results/supervised", exist_ok=True); T0 = time.time()
for ds in sys.argv[1].split(","):
    path, pp, SPL, kw = DS[ds]
    Hd = RV.load_hourly(path); pairs = json.load(open(pp)); rows, info = [], {}
    for name, (up, dn) in pairs.items():
        idx, U, D = SV.filled_pair(Hd, up, dn); tz = idx.tz
        T = {k: tuple(pd.Timestamp(v, tz=tz) for v in SPL[k]) for k in ("train", "val", "test")}
        WTs = RV.make_windows(Hd, {name: pairs[name]}, "test", **kw)
        pos = pd.Series(np.arange(len(idx)), index=idx)
        pte = np.array([int(pos[pd.Timestamp(w["meta"]["origin"])]) for w in WTs])
        pva = SV.valid_positions(idx, U, D, *T["val"])
        for hist in ("full", "short"):
            a = T["train"][0] if hist == "full" else T["train"][1] - pd.Timedelta(days=SHORT_DAYS.get(ds, 90))
            ptr = SV.valid_positions(idx, U, D, a, T["train"][1])
            for cov in (True, False):
                Xtr, Ytr, _ = SV.design(idx, U, D, ptr, use_cov=cov); Xva, Yva, _ = SV.design(idx, U, D, pva, use_cov=cov)
                Xte, Yte, bte = SV.design(idx, U, D, pte, use_cov=cov)
                assert np.allclose(Yte + bte[:, None], np.stack([w["y"][512:] for w in WTs]), atol=1e-3)
                for mdl, fit in (("arx", SV.fit_ridge), ("lgbm", SV.fit_lgbm)):
                    f, inf = fit(Xtr, Ytr, Xva, Yva)
                    ms = mase_windows(f(Xte) + bte[:, None], WTs, 512, 24)
                    arm = f"{mdl}_{hist}" + ("" if cov else "_nocov")
                    info[f"{name}/{arm}"] = {"n_train": int(len(ptr)), **{k: (v if k != "iters" else int(np.median(v))) for k, v in inf.items()}}
                    for i, w in enumerate(WTs):
                        rows.append({"set": f"{ds}_test", "arm": arm, "cell": name, "id": w["meta"]["origin"], "event": w["meta"]["event"], "mase": float(ms[i])})
                    print(f"[{(time.time()-T0)/60:5.1f} min] {ds} {name} {arm} n_train={len(ptr)} GM={np.exp(np.log(ms).mean()):.4f}", flush=True)
        pd.DataFrame(rows).to_csv(f"results/supervised/{ds}_rows.csv", index=False)
        json.dump(info, open(f"results/supervised/{ds}_info.json", "w"), indent=1)
print("DONE")
