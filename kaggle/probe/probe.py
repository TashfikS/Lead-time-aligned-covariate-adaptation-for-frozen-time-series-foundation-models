# API probe for four frozen TSFM backbones with covariates (Kaggle T4, internet on).
import os, sys, json, time, subprocess, traceback, inspect
T0 = time.time()
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./probe_out"
os.makedirs(OUT, exist_ok=True)
REP = {"steps": {}}
def log(*a):
    print(f"[{(time.time()-T0)/60:5.1f}m]", *a, flush=True)
def dump():
    json.dump(REP, open(os.path.join(OUT, "probe_report.json"), "w"), indent=1, default=str)
def pip(*args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], capture_output=True, text=True)
    return r.returncode, (r.stdout[-1500:] + r.stderr[-3000:])
def step(name, fn):
    log("STEP", name)
    try:
        REP["steps"][name] = {"ok": True, "out": fn()}
    except Exception as e:
        REP["steps"][name] = {"ok": False, "err": repr(e), "tb": traceback.format_exc()[-4000:]}
        log("FAIL", name, repr(e)[:300])
    dump()

import numpy as np
rng = np.random.default_rng(0)
C, H, TAU = 512, 24, 24
# base_t drives the target; the covariate is base shifted ahead by TAU, so y_t = 2*cov_{t-TAU} + noise.
base = np.convolve(rng.standard_normal(C + H + TAU + 50), np.ones(5) / 5, mode="same")[: C + H + TAU]
yt = 2.0 * base[: C + H] + 0.1 * rng.standard_normal(C + H)
cov = base[TAU: TAU + C + H]
shifted_full = np.concatenate([np.full(TAU, np.nan), cov[:-TAU]])   # = base_t for t >= TAU
truth = yt[C:C + H]
REP["toy"] = "y_t = 2*cov_{t-24} + noise; C=512 H=24; the oracle-shifted covariate is known over the horizon"
def mae(a, b): return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))

def env_info():
    import torch
    return {"py": sys.version, "torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "ngpu": torch.cuda.device_count()}
step("env", env_info)

# ---------------- Chronos-2 ----------------
def chronos():
    rc, msg = pip("chronos-forecasting==2.3.2")
    import torch, chronos
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda", torch_dtype=torch.float32)
    m = pipe.model.eval()
    res = {"pip_rc": rc, "version": getattr(chronos, "__version__", "?"),
           "predict_sig": str(inspect.signature(pipe.predict))[:800],
           "fwd_sig": str(inspect.signature(m.forward))[:800],
           "quantiles": list(m.chronos_config.quantiles), "out_patch": m.chronos_config.output_patch_size,
           "cfg": {k: str(v)[:80] for k, v in vars(m.chronos_config).items()}}
    OP = m.chronos_config.output_patch_size
    def run(rows_ctx, rows_fut):
        c = torch.tensor(np.array(rows_ctx), dtype=torch.float32, device="cuda")
        npatch = int(np.ceil(H / OP))
        f = np.full((len(rows_ctx), npatch * OP), np.nan, np.float32)
        for i, r in enumerate(rows_fut):
            if r is not None: f[i, :H] = r
        gid = torch.zeros(len(rows_ctx), dtype=torch.long, device="cuda")
        with torch.inference_mode():
            o = m(context=c, group_ids=gid, future_covariates=torch.tensor(f, device="cuda"), num_output_patches=npatch)
        qi = list(m.chronos_config.quantiles).index(0.5)
        return o.quantile_preds[0, qi, :H].float().cpu().numpy()
    sh_ctx = np.nan_to_num(shifted_full[:C], nan=0.0)
    res["mae"] = {"uni": mae(run([yt[:C]], [None]), truth),
                  "raw_past": mae(run([yt[:C], cov[:C]], [None, None]), truth),
                  "oracle_past_only": mae(run([yt[:C], sh_ctx], [None, None]), truth),
                  "oracle_future_known": mae(run([yt[:C], sh_ctx], [None, shifted_full[C:C + H]]), truth)}
    c = torch.tensor(np.array([yt[:C], cov[:C]]), dtype=torch.float32, device="cuda", requires_grad=True)
    gid = torch.zeros(2, dtype=torch.long, device="cuda")
    f = torch.full((2, 2 * OP), float("nan"), device="cuda")
    o = m(context=c, group_ids=gid, future_covariates=f, num_output_patches=2)
    o.quantile_preds[0].sum().backward()
    res["grad_norm_cov_row"] = float(c.grad[1].norm())
    return res
step("chronos2", chronos)

# ---------------- TimesFM-2.5 ----------------
def timesfm_probe():
    out = {}
    rc, msg = pip("timesfm[torch,xreg]")
    out["pip_pypi_rc"] = rc; out["pip_pypi_msg"] = msg[-600:]
    try:
        import timesfm
        has = hasattr(timesfm, "TimesFM_2p5_200M_torch")
    except Exception as e:
        has = False; out["import_err"] = repr(e)
    if not has:
        rc2, msg2 = pip("timesfm[torch,xreg] @ git+https://github.com/google-research/timesfm.git")
        out["pip_git_rc"] = rc2; out["pip_git_msg"] = msg2[-800:]
        for k in list(sys.modules):
            if k.startswith("timesfm"): del sys.modules[k]
        import timesfm
    out["version"] = getattr(timesfm, "__version__", "?")
    out["dir"] = [d for d in dir(timesfm) if not d.startswith("_")][:60]
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    out["ForecastConfig_sig"] = str(inspect.signature(timesfm.ForecastConfig))[:1500]
    model.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=256, normalize_inputs=True,
                                         use_continuous_quantile_head=True, force_flip_invariance=True,
                                         infer_is_positive=False, fix_quantile_crossing=True, return_backcast=True))
    pt, qt = model.forecast(horizon=H, inputs=[yt[:C]])
    out["uni_mae"] = mae(np.asarray(pt)[0][:H], truth)
    out["fwc_sig"] = str(inspect.signature(model.forecast_with_covariates))[:1500]
    def fwc(covarr):
        return model.forecast_with_covariates(inputs=[yt[:C]], dynamic_numerical_covariates={"x": [covarr]},
                                              xreg_mode="xreg + timesfm", ridge=0.0, force_on_cpu=False)
    r = fwc(np.concatenate([cov[:C], np.repeat(cov[C - 1], H)]))
    out["fwc_return_type"] = str(type(r)); out["fwc_len"] = len(r) if hasattr(r, "__len__") else None
    out["fwc_elem_types"] = [str(type(e)) for e in r][:4] if hasattr(r, "__len__") else None
    pt_raw = np.asarray(r[0][0]).reshape(-1)[:H]
    r2 = fwc(np.nan_to_num(shifted_full[:C + H], nan=0.0))
    pt_or = np.asarray(r2[0][0]).reshape(-1)[:H]
    out["mae"] = {"uni": out["uni_mae"], "raw_cov_future_held_last": mae(pt_raw, truth),
                  "oracle_future_known": mae(pt_or, truth)}
    return out
step("timesfm25", timesfm_probe)

# ---------------- TabPFN-TS ----------------
def tabpfn_probe():
    out = {}
    rc, msg = pip("tabpfn-time-series")
    out["pip_rc"] = rc; out["pip_msg"] = msg[-600:]
    import tabpfn_time_series as tts
    out["version"] = getattr(tts, "__version__", "?")
    out["dir"] = [d for d in dir(tts) if not d.startswith("_")][:80]
    import pandas as pd
    ts = pd.date_range("2020-01-01", periods=C + H, freq="h")
    ctx = pd.DataFrame({"item_id": "a", "timestamp": ts[:C], "target": yt[:C], "xs": np.nan_to_num(shifted_full[:C], nan=0.0)})
    fut = pd.DataFrame({"item_id": "a", "timestamp": ts[C:], "xs": shifted_full[C:C + H]})
    out["pipe_sig"] = str(inspect.signature(tts.TabPFNTSPipeline))[:1200]
    out["modes"] = [m for m in dir(tts.TabPFNMode) if not m.startswith("_")] if hasattr(tts, "TabPFNMode") else None
    p = tts.TabPFNTSPipeline(tabpfn_mode=tts.TabPFNMode.LOCAL) if hasattr(tts, "TabPFNMode") else tts.TabPFNTSPipeline()
    out["predict_df_sig"] = str(inspect.signature(p.predict_df))[:1200]
    pr = p.predict_df(context_df=ctx, future_df=fut, prediction_length=H)
    out["pred_cols"] = [str(c) for c in pr.columns][:20]
    out["pred_head"] = pr.head(3).to_string()[:800]
    col = "target" if "target" in pr.columns else ("predictions" if "predictions" in pr.columns else pr.columns[-1])
    out["mae_oracle_future_known"] = mae(pr[col].values[:H], truth)
    fut0 = fut.copy(); fut0["xs"] = np.nan
    pr0 = p.predict_df(context_df=ctx.assign(xs=cov[:C]), future_df=fut0, prediction_length=H)
    out["mae_raw_pastonly_nanfuture"] = mae(pr0[col].values[:H], truth)
    return out
step("tabpfn_ts", tabpfn_probe)

# ---------------- Moirai 1.1 (uni2ts) ----------------
def moirai_probe():
    out = {}
    rc, msg = pip("--no-deps", "uni2ts")
    out["pip_rc"] = rc; out["pip_msg"] = msg[-600:]
    rc2, msg2 = pip("gluonts", "jaxtyping", "einops", "hydra-core", "lightning", "huggingface_hub", "safetensors")
    out["pip2_rc"] = rc2; out["pip2_msg"] = msg2[-600:]
    import torch
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    out["fc_sig"] = str(inspect.signature(MoiraiForecast))[:1500]
    out["fwd_sig"] = str(inspect.signature(MoiraiForecast.forward))[:1500]
    mod = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base")
    def run(past_cov, fut_cov_known):
        fd = 1 if fut_cov_known is not None else 0
        pd_ = 1 if past_cov is not None else 0
        f = MoiraiForecast(module=mod, prediction_length=H, context_length=C, patch_size=32, num_samples=50,
                           target_dim=1, feat_dynamic_real_dim=fd, past_feat_dynamic_real_dim=pd_).cuda().eval()
        kw = dict(past_target=torch.tensor(yt[:C], dtype=torch.float32)[None, :, None].cuda(),
                  past_observed_target=torch.ones(1, C, 1, dtype=torch.bool).cuda(),
                  past_is_pad=torch.zeros(1, C, dtype=torch.bool).cuda())
        if fd:
            kw["feat_dynamic_real"] = torch.tensor(fut_cov_known, dtype=torch.float32)[None, :, None].cuda()
            kw["observed_feat_dynamic_real"] = torch.ones(1, C + H, 1, dtype=torch.bool).cuda()
        if pd_:
            kw["past_feat_dynamic_real"] = torch.tensor(past_cov, dtype=torch.float32)[None, :, None].cuda()
            kw["past_observed_feat_dynamic_real"] = torch.ones(1, C, 1, dtype=torch.bool).cuda()
        with torch.inference_mode():
            s = f(**kw)
        return s.median(dim=1).values.reshape(-1)[:H].float().cpu().numpy()
    out["mae"] = {"uni": mae(run(None, None), truth), "raw_past": mae(run(cov[:C], None), truth),
                  "oracle_future_known": mae(run(None, np.nan_to_num(shifted_full[:C + H], nan=0.0)), truth)}
    return out
step("moirai11", moirai_probe)

REP["elapsed_min"] = (time.time() - T0) / 60
dump()
log("DONE")
