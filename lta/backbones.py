"""Unified covariate interface for frozen TSFMs.

forecast(y_ctx, cov_ctx, cov_fut) -> point (N,H)
  y_ctx  : (N, C) float, target context
  cov_ctx: (N, k, C) or None, covariate values over the context (NaN = missing)
  cov_fut: (N, k, H) or None, covariate values over the horizon (NaN = unknown)
A covariate whose future is entirely NaN is a past-only covariate.
Each wrapper documents how it treats NaN, since the backbones' native interfaces differ.
"""
import math
import numpy as np


def _ffill_bfill(a):
    """Fill NaN along the last axis: forward fill, then back fill leading NaNs. Rows of all NaN -> 0."""
    a = np.array(a, dtype=np.float32, copy=True)
    flat = a.reshape(-1, a.shape[-1])
    for r in range(flat.shape[0]):
        v = flat[r]
        ok = np.isfinite(v)
        if not ok.any():
            v[:] = 0.0
            continue
        idx = np.where(ok, np.arange(len(v)), 0)
        np.maximum.accumulate(idx, out=idx)
        v[:] = v[idx]
        first = np.argmax(ok)
        v[:first] = v[first]
    return flat.reshape(a.shape)


class Chronos2:
    """amazon/chronos-2 via its low-level forward. Native NaN support, used through explicit masks so that
    gradients stay finite (NaN values never enter autograd). Each series is one group: row 0 = target, rows
    1..k = covariates. Covariate rows carry their known future values; the target row's future is masked."""
    name = "chronos2"

    def __init__(self, model_id="amazon/chronos-2", device="cuda", batch_rows=256):
        import torch
        from chronos import Chronos2Pipeline
        self.torch = torch
        self.pipe = Chronos2Pipeline.from_pretrained(model_id, device_map=device, torch_dtype=torch.float32)
        self.m = self.pipe.model.eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.device = device
        self.OP = self.m.chronos_config.output_patch_size
        self.Q = list(self.m.chronos_config.quantiles)
        self.qi = self.Q.index(0.5)
        self.batch_rows = batch_rows

    def forward_tensors(self, y_ctx, cov_ctx, cov_fut, H):
        """Differentiable core. y_ctx (N,C) tensor; cov_ctx (N,k,C) tensor or None; cov_fut (N,k,H) or None.
        Returns quantile predictions (N, Q, H) for the target rows."""
        torch = self.torch
        N, C = y_ctx.shape
        k = 0 if cov_ctx is None else cov_ctx.shape[1]
        g = 1 + k
        npatch = math.ceil(H / self.OP)
        Hp = npatch * self.OP
        ctx = y_ctx[:, None, :] if k == 0 else torch.cat([y_ctx[:, None, :], cov_ctx], 1)   # (N,g,C)
        ctx = ctx.reshape(N * g, C)
        fut = torch.full((N, g, Hp), float("nan"), device=ctx.device, dtype=ctx.dtype)
        if k:
            fut = torch.cat([fut[:, :1], torch.cat([cov_fut, fut[:, 1:, H:]], 2)], 1)
        fut = fut.reshape(N * g, Hp)
        cm = torch.isfinite(ctx).to(ctx.dtype)
        fm = torch.isfinite(fut).to(ctx.dtype)
        gid = torch.arange(N, device=ctx.device).repeat_interleave(g)
        o = self.m(context=torch.nan_to_num(ctx), context_mask=cm, group_ids=gid,
                   future_covariates=torch.nan_to_num(fut), future_covariates_mask=fm, num_output_patches=npatch)
        return o.quantile_preds[0::g, :, :H]

    def forecast(self, y_ctx, cov_ctx=None, cov_fut=None, H=None, return_quantiles=False):
        torch = self.torch
        N, C = y_ctx.shape
        H = cov_fut.shape[2] if cov_fut is not None else H
        k = 0 if cov_ctx is None else cov_ctx.shape[1]
        bs = max(1, self.batch_rows // (1 + k))
        outs = []
        with torch.inference_mode():
            for s in range(0, N, bs):
                e = min(N, s + bs)
                yc = torch.tensor(y_ctx[s:e], dtype=torch.float32, device=self.device)
                cc = None if k == 0 else torch.tensor(cov_ctx[s:e], dtype=torch.float32, device=self.device)
                cf = None if k == 0 else torch.tensor(cov_fut[s:e], dtype=torch.float32, device=self.device)
                outs.append(self.forward_tensors(yc, cc, cf, H).float().cpu().numpy())
        q = np.concatenate(outs, 0)
        return (q[:, self.qi], q) if return_quantiles else q[:, self.qi]


class TimesFM25:
    """google/timesfm-2.5-200m-pytorch. Covariates only through XReg (in-context linear regression on
    same-time covariate values), which requires values over context+horizon without NaN. NaN in the
    context are filled forward/backward; NaN in the horizon (unknown future) are held at the last known
    value. Without covariates the plain forecast() is used."""
    name = "timesfm25"

    def __init__(self, max_context=1024, max_horizon=256, model_id="google/timesfm-2.5-200m-pytorch"):
        import timesfm
        self.tfm = timesfm
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
        self.model.compile(timesfm.ForecastConfig(max_context=max_context, max_horizon=max_horizon,
                                                  normalize_inputs=True, use_continuous_quantile_head=True,
                                                  force_flip_invariance=True, infer_is_positive=False,
                                                  fix_quantile_crossing=True, return_backcast=True))

    def forecast(self, y_ctx, cov_ctx=None, cov_fut=None, H=None, return_quantiles=False):
        N, C = y_ctx.shape
        if cov_ctx is None:
            pt, qt = self.model.forecast(horizon=H, inputs=[r for r in y_ctx])
            # return_backcast=True (required by XReg) prepends the backcast: the horizon is the LAST H steps.
            return np.asarray(pt)[:, -H:]
        k = cov_ctx.shape[1]; H = cov_fut.shape[2]
        full = np.concatenate([cov_ctx, cov_fut], 2)            # (N,k,C+H)
        full = _ffill_bfill(full)
        # XReg solves ONE least-squares problem over all series passed in a call (shared coefficients), so each
        # independent series is fitted in its own call to obtain per-series in-context regression.
        preds = []
        for i in range(N):
            dyn = {f"x{j}": [full[i, j]] for j in range(k)}
            out, _ = self.model.forecast_with_covariates(inputs=[y_ctx[i]], dynamic_numerical_covariates=dyn,
                                                         xreg_mode="xreg + timesfm", ridge=0.0, force_on_cpu=False)
            preds.append(np.asarray(out[0]).reshape(-1)[:H])
        return np.stack(preds)


class TabPFNTS:
    """TabPFN-TS (local). Forecasting as tabular regression on (calendar/running-index features + covariates).
    Native NaN support in features: unknown future covariate values are passed as NaN."""
    name = "tabpfn_ts"

    def __init__(self, version="v2"):
        # TabPFN-TS (Hoo et al., 2025) was published on TabPFN v2; later checkpoints (v2.5+) are licence-gated
        # behind browser acceptance and cannot be fetched headless, so v2 weights are pinned explicitly.
        import tabpfn_time_series as tts
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        self.tts = tts
        mp = TabPFNRegressor.create_default_for_version(ModelVersion(version)).model_path
        self.p = tts.TabPFNTSPipeline(tabpfn_mode=tts.TabPFNMode.LOCAL, tabpfn_model_config={"model_path": mp},
                                      compile=False)

    def forecast(self, y_ctx, cov_ctx=None, cov_fut=None, H=None, freq="h"):
        import pandas as pd
        N, C = y_ctx.shape
        if cov_fut is not None:
            H = cov_fut.shape[2]
        ts = pd.date_range("2020-01-01", periods=C + H, freq=freq)
        ctx_rows, fut_rows = [], []
        for i in range(N):
            d = {"item_id": f"s{i}", "timestamp": ts[:C], "target": y_ctx[i]}
            f = {"item_id": f"s{i}", "timestamp": ts[C:]}
            if cov_ctx is not None:
                for j in range(cov_ctx.shape[1]):
                    d[f"x{j}"] = cov_ctx[i, j]
                    f[f"x{j}"] = cov_fut[i, j]
            ctx_rows.append(pd.DataFrame(d)); fut_rows.append(pd.DataFrame(f))
        pr = self.p.predict_df(context_df=pd.concat(ctx_rows), future_df=pd.concat(fut_rows))
        pr = pr.reset_index() if "item_id" not in pr.columns else pr
        col = "target" if "target" in pr.columns else ("0.5" if "0.5" in pr.columns else 0.5)
        out = np.zeros((N, H), np.float32)
        for i in range(N):
            out[i] = pr[pr["item_id"] == f"s{i}"][col].values[:H]
        return out


class Moirai11:
    """Salesforce/moirai-1.1-R-base via uni2ts. Known-future covariates via feat_dynamic_real with an observed
    mask (unknown horizon values -> observed=False); a covariate whose horizon is entirely unknown is passed
    as past_feat_dynamic_real. Median of 100 samples."""
    name = "moirai11"

    def __init__(self, size="base", device="cuda", num_samples=100):
        import torch
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        self.torch = torch; self.MF = MoiraiForecast
        self.mod = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}")
        self.device = device; self.ns = num_samples

    def forecast(self, y_ctx, cov_ctx=None, cov_fut=None, H=None, bs=64):
        torch = self.torch
        N, C = y_ctx.shape
        if cov_fut is not None:
            H = cov_fut.shape[2]
        k = 0 if cov_ctx is None else cov_ctx.shape[1]
        if k:
            known = np.isfinite(cov_fut).any(2)                    # (N,k)
            fut_idx = [j for j in range(k) if known[:, j].any()]
            past_idx = [j for j in range(k) if not known[:, j].any()]
        else:
            fut_idx, past_idx = [], []
        f = self.MF(module=self.mod, prediction_length=H, context_length=C, patch_size=32, num_samples=self.ns,
                    target_dim=1, feat_dynamic_real_dim=len(fut_idx), past_feat_dynamic_real_dim=len(past_idx))
        f = f.to(self.device).eval()
        outs = []
        for s in range(0, N, bs):
            e = min(N, s + bs); n = e - s
            kw = dict(past_target=torch.tensor(y_ctx[s:e], dtype=torch.float32)[:, :, None].to(self.device),
                      past_observed_target=torch.ones(n, C, 1, dtype=torch.bool, device=self.device),
                      past_is_pad=torch.zeros(n, C, dtype=torch.bool, device=self.device))
            if fut_idx:
                v = np.concatenate([cov_ctx[s:e][:, fut_idx], cov_fut[s:e][:, fut_idx]], 2).transpose(0, 2, 1)
                kw["feat_dynamic_real"] = torch.tensor(np.nan_to_num(v), dtype=torch.float32, device=self.device)
                kw["observed_feat_dynamic_real"] = torch.tensor(np.isfinite(v), device=self.device)
            if past_idx:
                v = cov_ctx[s:e][:, past_idx].transpose(0, 2, 1)
                kw["past_feat_dynamic_real"] = torch.tensor(np.nan_to_num(v), dtype=torch.float32, device=self.device)
                kw["past_observed_feat_dynamic_real"] = torch.tensor(np.isfinite(v), device=self.device)
            with torch.inference_mode():
                smp = f(**kw)                                       # (n, ns, H)
            outs.append(smp.median(dim=1).values.reshape(n, -1)[:, :H].float().cpu().numpy())
        return np.concatenate(outs, 0)
