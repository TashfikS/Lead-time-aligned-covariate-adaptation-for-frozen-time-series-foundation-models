"""CoRA (Qin et al., arXiv:2510.12681, Sec. 3) instantiated on a frozen Chronos-2.

E_target : encoder states at the future-patch tokens of the univariate target run (input of the Chronos-2 head)
E_cov    : encoder state at the last context patch of the covariate run univariately (latest-known values)
H        = softmax(W_GC) . (E_cov W + b)          (W, b zero-initialised; with one covariate the gate is the identity)
gamma, beta, alpha = MLP(H)                       (adaLN-Zero: output layer zero-initialised)
forecast = (1 + alpha) * Head(gamma + (1 + beta) * E_target), alpha per horizon step, in the normalised scale.
At initialisation the forecast equals the frozen univariate Chronos-2 forecast exactly.
"""
import math
import torch
import torch.nn as nn


@torch.no_grad()
def embed(m, y_ctx, x_ctx, H, bs=256):
    """Return E_target (N,n,D), E_cov (N,1,D), loc, scale for target contexts y_ctx (N,C) and covariate x_ctx (N,C)."""
    OP = m.chronos_config.output_patch_size; n = math.ceil(H / OP)
    Et, Ec, L, S = [], [], [], []
    for s in range(0, len(y_ctx), bs):
        y = y_ctx[s:s + bs]; x = x_ctx[s:s + bs]
        out, (loc, sc), _, nctx = m.encode(context=y, num_output_patches=n)
        Et.append(out[0][:, -n:]); L.append(loc); S.append(sc)
        outx, _, _, nctx_x = m.encode(context=x, num_output_patches=1)
        Ec.append(outx[0][:, nctx_x - 1:nctx_x])
    return torch.cat(Et), torch.cat(Ec), torch.cat(L), torch.cat(S)


class CoRAHead(nn.Module):
    def __init__(self, m, H=24, n_cov=1):
        super().__init__()
        self.m = m; D = m.model_dim; self.H = H; self.D = D
        self.OP = m.chronos_config.output_patch_size; self.n = math.ceil(H / self.OP)
        self.Q = m.num_quantiles
        self.proj = nn.Linear(D, D); nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        self.gc = nn.Parameter(torch.zeros(n_cov))
        self.mlp = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, 2 * D + H))
        nn.init.zeros_(self.mlp[2].weight); nn.init.zeros_(self.mlp[2].bias)

    def forward(self, Et, Ec):
        """Et (B,n,D), Ec (B,N,D) -> normalised quantile predictions (B, Q, n*OP)."""
        Hh = (torch.softmax(self.gc, 0)[None, :, None] * self.proj(Ec)).sum(1)
        g, b, a = torch.split(self.mlp(Hh), [self.D, self.D, self.H], -1)
        z = self.m.output_patch_embedding(g[:, None] + (1 + b[:, None]) * Et)          # (B,n,Q*OP)
        B = z.shape[0]
        q = z.reshape(B, self.n, self.Q, self.OP).permute(0, 2, 1, 3).reshape(B, self.Q, self.n * self.OP)
        a_full = torch.cat([a, torch.zeros(B, self.n * self.OP - self.H, device=a.device)], -1)
        return (1 + a_full[:, None, :]) * q

    def unscale(self, q, loc, scale):
        B = q.shape[0]
        flat = self.m.instance_norm.inverse(q.reshape(B, -1), (loc, scale))
        return flat.reshape(B, self.Q, -1)[..., :self.H]


def pinball(qn, yn, quantiles):
    """qn (B,Q,>=H) normalised predictions; yn (B,H) normalised targets."""
    H = yn.shape[1]; q = qn[..., :H]; t = quantiles.view(1, -1, 1)
    return (2 * torch.abs((yn[:, None] - q) * ((yn[:, None] <= q).float() - t))).mean()
