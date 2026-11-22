"""Training the Lead-Time Aligner through a frozen Chronos-2 (backbone-in-the-loop quantile loss)."""
import math, time, json
import numpy as np
import torch
from .synth import make_task
from .aligner import LeadTimeAligner

TRAIN_FAMS = ["linear", "periodic", "pulse", "nonlinear", "tvlag", "null"]


def sample_spec(rng):
    fam = TRAIN_FAMS[rng.integers(len(TRAIN_FAMS))]
    if fam == "tvlag":
        tau0 = int(rng.integers(30, 121)); delta = int(rng.integers(4, min(30, tau0 - 1) + 1))
    else:
        tau0 = int(rng.integers(0, 121)); delta = None
    return fam, tau0, delta


def build_pool(n, split, C, H, seed=0):
    """Deterministic pool of synthetic tasks. Index i gets spec from SeedSequence((seed, i))."""
    pool = []
    for i in range(n):
        r = np.random.default_rng(np.random.SeedSequence((991, seed, i)))
        fam, tau0, delta = sample_spec(r)
        T = make_task(fam, tau0, split, i, C=C, H=H, delta=delta)
        pool.append(T)
    return pool


AUX_FAMS = ("linear", "periodic", "nonlinear", "tvlag")   # families with a point lead time


def aux_targets(tasks, C, H, K):
    """Median true lag over the context (CE target), oracle-aligned covariate (C+H), and an aux mask."""
    lag = np.array([int(np.clip(np.median(t["tau"][:C]), 0, K)) for t in tasks])
    xo = np.stack([t["xo"] for t in tasks]).astype(np.float32)[:, None, :]
    m = np.array([t["meta"]["family"] in AUX_FAMS for t in tasks])
    return lag, xo, m


def batch_arrays(tasks, C, H):
    y = np.stack([t["y"] for t in tasks]).astype(np.float32)        # (B, C+H)
    x = np.stack([t["x"][:C] for t in tasks]).astype(np.float32)[:, None, :]
    return y, x


def lta_covariates(xa, x, C):
    """Backbone covariate rows: [aligned (context + known horizon), raw (past-only)]."""
    B, k, N = xa.shape
    H = N - C
    cov_ctx = torch.cat([xa[:, :, :C], x], 1)
    cov_fut = torch.cat([xa[:, :, C:], torch.full((B, k, H), float("nan"), device=x.device)], 1)
    return cov_ctx, cov_fut


def quantile_loss(q, y, levels):
    """q (B,Q,H), y (B,H), levels list -> mean pinball loss."""
    lv = torch.tensor(levels, device=q.device, dtype=q.dtype)[None, :, None]
    d = y[:, None, :] - q
    return torch.maximum(lv * d, (lv - 1) * d).mean(-1).mean(-1)   # (B,)


def train_lta(backbone, pool, val_fn, init_state=None, C=512, H=24, K=168, steps=4000, bs=32, lr=2e-3, wd=1e-4,
              dynamic=True, eval_every=250, log=print, device="cuda", seed=0, time_budget_min=None,
              aux_ce=1.0, aux_mse=1.0, ccf_init_scale=10.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lta = LeadTimeAligner(K=K, dynamic=dynamic, ccf_init_scale=ccf_init_scale).to(device)
    if init_state is not None:
        lta.load_state_dict(init_state)
    opt = torch.optim.AdamW(lta.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    levels = backbone.Q
    hist, best = [], {"val": float("inf"), "step": -1, "state": None}
    t0 = time.time()
    for step in range(steps + 1):
        if step % eval_every == 0:
            lta.eval()
            v = val_fn(lta)
            hist.append({"step": step, "val": v, "min": (time.time() - t0) / 60})
            if v < best["val"]:
                best = {"val": v, "step": step, "state": {k: t.detach().cpu().clone() for k, t in lta.state_dict().items()}}
            log(f"step {step} val {v:.4f} best {best['val']:.4f}@{best['step']} ccf_scale {lta.ccf_scale.item():.2f}")
            lta.train()
            if time_budget_min and (time.time() - t0) / 60 > time_budget_min:
                log("time budget reached"); break
        if step == steps:
            break
        idx = rng.integers(0, len(pool), bs)
        tasks = [pool[i] for i in idx]
        y, x = batch_arrays(tasks, C, H)
        y = torch.tensor(y, device=device); x = torch.tensor(x, device=device)
        yc, yf = y[:, :C], y[:, C:]
        scale = (yc[:, 24:] - yc[:, :-24]).abs().mean(-1).clamp(min=1e-3)      # seasonal-naive scale
        xa, aux = lta(yc, x, H, return_aux=True)
        cc, cf = lta_covariates(xa, x, C)
        q = backbone.forward_tensors(yc, cc, cf, H)
        loss_f = (quantile_loss(q, yf, levels) / scale).mean()
        loss = loss_f
        use_aux = (aux_ce > 0 or aux_mse > 0) and all("tau" in t for t in tasks)
        if use_aux:
            lag_t, xo_t, am = aux_targets(tasks, C, H, K)
            am_t = torch.tensor(am, device=device)
        if use_aux and am.any():
            lp = torch.log_softmax(aux["logits"][:, 0] / aux["T"][:, :1], -1)               # (B,K+1)
            ce = -lp.gather(1, torch.tensor(lag_t, device=device)[:, None]).squeeze(1)
            xo_ = torch.tensor(xo_t, device=device)
            xs_m, xs_s = x.mean(-1, keepdim=True), x.std(-1, keepdim=True) + 1e-6
            ok = torch.isfinite(xa) & torch.isfinite(xo_)
            se = torch.where(ok, ((torch.nan_to_num(xa) - xo_) / xs_s) ** 2, torch.zeros_like(xo_))
            mse = se.sum((1, 2)) / ok.sum((1, 2)).clamp(min=1)
            loss = loss + aux_ce * (ce * am_t).sum() / am_t.sum() + aux_mse * (mse * am_t).sum() / am_t.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lta.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0:
            hist.append({"step": step, "train_loss": float(loss.detach()), "forecast_loss": float(loss_f.detach())})
    lta.load_state_dict(best["state"])
    return lta, hist, best["step"]


@torch.no_grad()
def lta_forecast(lta, backbone, tasks, C, H, bs=64, device="cuda"):
    """Point forecasts of the frozen backbone (Chronos2 wrapper with forward_tensors) using LTA covariates."""
    out = []
    for s in range(0, len(tasks), bs):
        y, x = batch_arrays(tasks[s:s + bs], C, H)
        yc = torch.tensor(y[:, :C], device=device); xt = torch.tensor(x, device=device)
        xa = lta(yc, xt, H)
        cc, cf = lta_covariates(xa, xt, C)
        q = backbone.forward_tensors(yc, cc, cf, H)
        out.append(q[:, backbone.qi].float().cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def lta_aligned_numpy(lta, tasks, C, H, bs=64, device="cuda"):
    """Aligned covariate (N,1,C+H) with NaN where unknown -> for transfer to other backbones."""
    out = []
    for s in range(0, len(tasks), bs):
        y, x = batch_arrays(tasks[s:s + bs], C, H)
        xa = lta(torch.tensor(y[:, :C], device=device), torch.tensor(x, device=device), H)
        out.append(xa.float().cpu().numpy())
    return np.concatenate(out, 0)
