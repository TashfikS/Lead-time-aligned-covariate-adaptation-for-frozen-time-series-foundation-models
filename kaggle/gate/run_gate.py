# ---- Phase-0 headroom gate runner (protocol/P0_gate_protocol.md). CFG and lta/ are injected above. ----
import time, subprocess, traceback
T0 = time.time()
OUT = _W
def P(*a):
    print(f"PROGRESS [{(time.time()-T0)/60:6.1f} min]", *a, flush=True)
def pip(*args):
    if CFG.get("local") == "1": return 0
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], capture_output=True, text=True)
    P("pip", args[:3], "rc", r.returncode, r.stderr[-300:].replace("\n", " "))
    return r.returncode

BBS = CFG["backbones"].split(",")
N_PER = int(CFG.get("n_per_cell", 50))
N_TAB = int(CFG.get("n_per_cell_tabpfn", 20))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
if "chronos2" in BBS: pip("chronos-forecasting==2.3.2")
if "timesfm25" in BBS: pip("timesfm[torch,xreg]")
if "tabpfn_ts" in BBS: pip("tabpfn-time-series")
if "moirai11" in BBS:
    pip("--no-deps", "uni2ts")
    pip("gluonts", "jaxtyping", "einops", "hydra-core==1.3.2", "omegaconf==2.3.0", "lightning", "safetensors")

import numpy as np, pandas as pd
from scipy import stats
from lta.gate import build_tasks, arm_inputs, mase, ARMS
from lta import backbones as B

C, H = 512, 24
rows = []
summary = {"cfg": CFG, "backbones": {}}
def dump():
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "gate_per_series.csv"), index=False)
    summary["elapsed_min"] = (time.time() - T0) / 60
    json.dump(summary, open(os.path.join(OUT, "gate_summary.json"), "w"), indent=1, default=str)

for bb in BBS:
    n = N_TAB if bb == "tabpfn_ts" else N_PER
    tasks = build_tasks("dev", n, C, H)
    P("backbone", bb, "n_series", len(tasks))
    try:
        if bb == "chronos2": model = B.Chronos2(CFG.get("chronos_id", "amazon/chronos-2"), device=CFG.get("device", "cuda"))
        elif bb == "timesfm25": model = B.TimesFM25()
        elif bb == "tabpfn_ts": model = B.TabPFNTS("v2")
        elif bb == "moirai11": model = B.Moirai11("base", device="cuda")
        summary["backbones"][bb] = {"loaded": True, "arms": {}}
    except Exception as e:
        summary["backbones"][bb] = {"loaded": False, "err": repr(e), "tb": traceback.format_exc()[-3000:]}
        P("LOAD FAIL", bb, repr(e)[:300]); dump(); continue
    res = {}
    for arm in ARMS:
        t1 = time.time()
        try:
            y, cc, cf, lags = arm_inputs(tasks, arm, C, H)
            pred = model.forecast(y, cc, cf, H=H)
            assert pred.shape == (len(tasks), H) and np.isfinite(pred).all(), f"bad pred {pred.shape}"
            res[arm] = mase(pred, tasks, C, H)
            for i, t in enumerate(tasks):
                rows.append({"backbone": bb, "arm": arm, "cell": t["cell"], "index": t["meta"]["index"],
                             "mase": float(res[arm][i]), "lag_used": int(lags[i]) if lags is not None else -1,
                             "tau_at_C": int(t["tau"][C])})
            summary["backbones"][bb]["arms"][arm] = {"gm_mase": float(np.exp(np.log(res[arm]).mean())),
                                                     "sec": time.time() - t1}
            P(bb, arm, "GM-MASE", round(summary["backbones"][bb]["arms"][arm]["gm_mase"], 4), f"{time.time()-t1:.0f}s")
        except Exception as e:
            summary["backbones"][bb]["arms"][arm] = {"err": repr(e), "tb": traceback.format_exc()[-2000:]}
            P("ARM FAIL", bb, arm, repr(e)[:300])
        dump()
    del model
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
P("DONE"); dump()
