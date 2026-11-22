"""Regenerate all manuscript tables from result CSVs. Run from the project root: python paper/make_tables.py"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, ".")
from lta import analysis as AN

os.makedirs("paper/tables", exist_ok=True)
gmr = lambda a, b: float(np.exp((np.log(a) - np.log(b)).mean()))
gm = lambda a: float(np.exp(np.log(a).mean()))
f3 = lambda v: "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def wil(a, b, tol=1e-6):
    """Wilcoxon signed-rank on paired MASE; |differences| < tol (float32 noise of identical forecasts) are ties (zeros)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d[np.abs(d) < tol] = 0.0
    return stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0


def pfmt(p):
    if p is None or np.isnan(p):
        return "--"
    if p < 1e-10:
        return r"$<10^{-10}$"
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        return rf"${p / 10**e:.1f}\times10^{{{e}}}$"
    return f"{p:.3f}"


# ---------------- Table: gate ----------------
G = {"Chronos-2": ("results/gate_k1/gate_per_series.csv", "chronos2"),
     "TimesFM-2.5": ("results/gate_k3/gate_per_series.csv", "timesfm25"),
     "TabPFN-TS": ("results/gate_k2/gate_per_series.csv", "tabpfn_ts"),
     "Moirai-1.1": ("results/gate_k2/gate_per_series.csv", "moirai11")}
rows = []
for name, (f, key) in G.items():
    df = pd.read_csv(f); d = df[df.backbone == key].pivot_table(index=["cell", "index"], columns="arm", values="mase")
    if key == "timesfm25" and os.path.exists("results/timesfm_uni_fix/gate_uni.csv"):   # D12 correction
        fx = pd.read_csv("results/timesfm_uni_fix/gate_uni.csv").set_index(["cell", "index"])["mase_new"]
        assert fx.index.equals(d.index) or set(fx.index) == set(d.index)
        d["uni"] = fx.reindex(d.index).values
    ref = "uni" if gm(d["uni"]) < gm(d["raw"]) else "raw"
    g1 = gmr(d["oracle"], d[ref]); p1 = wil(d["oracle"], d[ref])
    g2 = gmr(d["randlag"], d["oracle"])
    best = np.minimum(d["ccf"], d["ccf_pw"])
    rows.append([name, len(d), gm(d["uni"]), gm(d["raw"]), gm(d["oracle_pastonly"]), gm(d["randlag"]), gm(d["ccf"]), gm(d["oracle"]),
                 ref, g1, p1, g2, (g1 <= 0.95 and p1 < 0.01 and g2 >= 1.05)])
T = pd.DataFrame(rows, columns=["bb", "n", "uni", "raw", "ctx", "rand", "ccf", "oracle", "ref", "g1", "p1", "g2", "passed"])
T.to_csv("paper/tables/tab_gate.csv", index=False)
L = [r"\begin{table}[t]\centering\small", r"\caption{Pre-registered headroom gate (dev split; GM-MASE). G1: GM[oracle/ref] $\le 0.95$ with Wilcoxon $p<0.01$, ref = better of no covariate and raw covariate. G2: GM[random lag/oracle] $\ge 1.05$. ``Context only'': true-lag shift in the context, horizon left missing.}",
     r"\label{tab:gate}", r"\begin{tabular}{lrrrrrrrrrrc}", r"\toprule",
     r"Backbone & $n$ & no cov. & raw & context only & random lag & CCF & oracle & G1 ratio & $p$ & G2 ratio & pass \\", r"\midrule"]
for r in T.itertuples():
    L.append(f"{r.bb} & {r.n} & {r.uni:.3f} & {r.raw:.3f} & {r.ctx:.3f} & {r.rand:.3f} & {r.ccf:.3f} & {r.oracle:.3f} & {r.g1:.3f} & {pfmt(r.p1)} & {r.g2:.3f} & {'yes' if r.passed else 'no'} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("paper/tables/tab_gate.tex", "w").write("\n".join(L))

# ---------------- Table: synthetic test ----------------
st = pd.read_csv("results/main_k1/synth_test_rows.csv")
extra = []
for d, t in [("abl_seed1", "lta_s1"), ("abl_seed2", "lta_s2"), ("abl_static", "lta_static"), ("abl_noccf", "lta_noccf"), ("abl_noaux", "lta_noaux")]:
    x = pd.read_csv(f"results/{d}/synth_test_rows.csv"); x = x[x.arm == "lta_zs"].copy(); x["arm"] = t; extra.append(x)
ws = pd.concat([st] + extra).pivot_table(index=["cell", "id"], columns="arm", values="mase", aggfunc="first")
fam = ws.index.get_level_values(0).str.extract(r"^([a-z]+)")[0].values
FAMS = [("linear", "fixed lag, linear"), ("periodic", "fixed lag, periodic"), ("nonlinear", "fixed lag, nonlinear"),
        ("pulse", "distributed response"), ("tvlag", "time-varying lag"), ("null", "uninformative"), ("ALL", "all")]
rows = []
for f, lab in FAMS:
    g = ws if f == "ALL" else ws[fam == f]
    seeds = [gmr(g[a], g["ccf"]) for a in ["lta_zs", "lta_s1", "lta_s2"]]
    rows.append([lab, len(g), gm(g["raw"]), gm(g["ccf"]), gm(g["top3"]), gm(g["gcc_phat"]), gm(g["oracle"]), gm(g["lta_zs"]),
                 gmr(g["oracle"], g["ccf"]), seeds[0], min(seeds), max(seeds), wil(g["lta_zs"], g["ccf"]),
                 gmr(g["lta_static"], g["ccf"]), gmr(g["lta_noccf"], g["ccf"]), gmr(g["lta_noaux"], g["ccf"])])
S = pd.DataFrame(rows, columns=["fam", "n", "raw", "ccf", "top3", "gcc", "oracle", "lta", "or_ccf", "lta_ccf", "lo", "hi", "p", "static", "noccf", "noaux"])
S.to_csv("paper/tables/tab_synth.csv", index=False)
L = [r"\begin{table}[t]\centering\small", r"\caption{Held-out synthetic test set (Chronos-2). Left: GM-MASE. Right: GM of per-series MASE ratios relative to CCF shifting; LTA range over three seeds; $p$: Wilcoxon, LTA seed 0 vs CCF.}",
     r"\label{tab:synth}", r"\begin{tabular}{lrrrrrrr|rrr}", r"\toprule",
     r" & & \multicolumn{6}{c|}{GM-MASE} & \multicolumn{3}{c}{ratio to CCF} \\",
     r"Family & $n$ & raw & CCF & top-3 & GCC-PHAT & oracle & LTA & oracle & LTA (seeds) & $p$ \\", r"\midrule"]
for r in S.itertuples():
    L.append(f"{r.fam} & {r.n} & {r.raw:.3f} & {r.ccf:.3f} & {r.top3:.3f} & {r.gcc:.3f} & {r.oracle:.3f} & {r.lta:.3f} & {r.or_ccf:.3f} & {r.lo:.3f}--{r.hi:.3f} & {pfmt(r.p)} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("paper/tables/tab_synth.tex", "w").write("\n".join(L))
print("synthetic + gate tables written")

# ---------------- RiverLead (Chronos-2) ----------------
def lta_rows(path, tag, split="river_test"):
    x = pd.read_csv(path); x = x[(x.arm == "lta_zs") & (x.set == split)].copy(); x["arm"] = tag; return x
rr = pd.read_csv("results/main_k1/river_rows.csv")
parts = [rr[rr.set == "river_test"], pd.read_csv("results/ft_k1/ft_rows.csv"), pd.read_csv("results/ft_lora/ft_rows.csv"),
         lta_rows("results/abl_seed1/river_rows.csv", "lta_s1"), lta_rows("results/abl_seed2/river_rows.csv", "lta_s2"),
         lta_rows("results/abl_static/river_rows.csv", "lta_static"), lta_rows("results/abl_noccf/river_rows.csv", "lta_noccf"),
         lta_rows("results/abl_noaux/river_rows.csv", "lta_noaux")]
RT = pd.concat(parts, ignore_index=True)
RT.loc[RT.arm == "lta_zs", "arm"] = "lta_s0"
WT = AN.wide(RT); WE = AN.wide(RT[RT.event == True])
VAL = AN.wide(rr[rr.set == "river_val"])
CLS = ["ccf", "ccf_pw", "gcc_phat", "top3", "global", "binned"]
best_r, val_scores = AN.best_on_val(VAL, CLS)
pooled = lambda w, a: float(np.exp(np.log(w[a]).groupby(level=0).mean().mean()))
ARMS_R = [("uni", "no covariate"), ("raw", "raw past covariate"), ("randlag", "random lag (control)"), ("gcc_phat", "GCC-PHAT"),
          ("ccf", "CCF"), ("global", "global train lag"), ("ccf_pw", "CCF, prewhitened"), ("top3", "top-3 CCF peaks"),
          ("binned", "flow-binned train lag"), ("lta_s0", "LTA, seed 0"), ("lta_s1", "LTA, seed 1"), ("lta_s2", "LTA, seed 2"),
          ("ft_full_raw", "fine-tuned Chronos-2 (full), raw"), ("ft_lora_raw", "fine-tuned Chronos-2 (LoRA), raw")]
def cmp_cells(w, a, ref):
    if a == ref: return ("--", "--", "--")
    c = AN.compare(w, a, ref)
    return (f"{c['pooled_ratio']:.3f}", f"[{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]", f"{c['pairs_better']}/{c['n_pairs']}")
rows = []; L = [r"\begin{table}[t]\centering\small",
    rf"\caption{{RiverLead test split (Chronos-2 backbone; 9 pairs, \num{{{len(WT)}}} windows). GM-MASE pooled over pairs; ratios with 7-day block-bootstrap 95\% intervals and number of pairs improved. The best classical arm on validation was {best_r.replace('_', chr(92)+'_')}. Fine-tuned rows modify the backbone and use the training split.}}",
    r"\label{tab:river}", r"\begin{tabular}{lrlllllr}", r"\toprule",
    r"Arm & GM-MASE & \multicolumn{3}{c}{ratio to raw} & \multicolumn{3}{c}{ratio to best classical} \\", r"\midrule"]
for a, lab in ARMS_R:
    r1 = cmp_cells(WT, a, "raw"); r2 = cmp_cells(WT, a, best_r); ev = cmp_cells(WE, a, "raw")
    rows.append([a, lab, pooled(WT, a), *r1, *r2, ev[0]])
    L.append(f"{lab} & {pooled(WT, a):.3f} & {r1[0]} & {r1[1]} & {r1[2]} & {r2[0]} & {r2[1]} & {r2[2]} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
L[L.index(r"Arm & GM-MASE & \multicolumn{3}{c}{ratio to raw} & \multicolumn{3}{c}{ratio to best classical} \\")] = \
    r"Arm & GM-MASE & \multicolumn{3}{c}{ratio to raw} & \multicolumn{3}{c}{ratio to best classical} \\"
L[3] = r"\begin{tabular}{lrrlrrlr}"
open("paper/tables/tab_river.tex", "w").write("\n".join(L))
pd.DataFrame(rows, columns=["arm", "label", "gm", "r_raw", "ci_raw", "pairs_raw", "r_best", "ci_best", "pairs_best", "event_r_raw"]).to_csv("paper/tables/tab_river.csv", index=False)
json.dump({"best_classical_val": best_r, "val_scores": val_scores}, open("paper/tables/river_selection.json", "w"), indent=1)

# ---------------- Ablations (synthetic tvlag/pulse/all + river) ----------------
rows = []
for a, lab in [("lta_zs", "full model (seed 0)"), ("lta_static", "no state-dependent offset"), ("lta_noccf", "no CCF initialisation"), ("lta_noaux", "no auxiliary losses")]:
    ra = "lta_s0" if a == "lta_zs" else a
    sy = {f: gmr(ws[fam == f][a], ws[fam == f]["ccf"]) for f in ["tvlag", "pulse", "nonlinear"]}
    c = AN.compare(WT, ra, best_r)
    rows.append([lab, sy["tvlag"], sy["pulse"], sy["nonlinear"], gmr(ws[a], ws["ccf"]), c["pooled_ratio"], c["ci_lo"], c["ci_hi"], c["pairs_better"]])
A = pd.DataFrame(rows, columns=["variant", "tv", "pulse", "nl", "all", "river", "lo", "hi", "pairs"]); A.to_csv("paper/tables/tab_ablation.csv", index=False)
L = [r"\begin{table}[t]\centering\small", rf"\caption{{Ablations (Chronos-2). Synthetic: GM ratio to CCF shifting on the test set. RiverLead: pooled ratio to the best classical arm ({best_r.replace('_', chr(92)+'_')}) with block-bootstrap 95\% interval and pairs improved (of 9).}}",
     r"\label{tab:ablation}", r"\begin{tabular}{lrrrrrlr}", r"\toprule",
     r" & \multicolumn{4}{c}{synthetic, ratio to CCF} & \multicolumn{3}{c}{RiverLead, ratio to best classical} \\",
     r"Variant & time-varying & distributed & nonlinear & all & ratio & 95\% CI & pairs \\", r"\midrule"]
for r in A.itertuples():
    L.append(f"{r.variant} & {r.tv:.3f} & {r.pulse:.3f} & {r.nl:.3f} & {r.all:.3f} & {r.river:.3f} & [{r.lo:.3f}, {r.hi:.3f}] & {r.pairs}/9 \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("paper/tables/tab_ablation.tex", "w").write("\n".join(L))
print("river + ablation tables written; best classical on val:", best_r)

# ---------------- AirLead ----------------
AR = pd.read_csv("results/air_chronos/air_chronos2_rows.csv")
AWT, AWV = AN.wide(AR[AR.set == "air_test"]), AN.wide(AR[AR.set == "air_val"])
best_a, va_scores = AN.best_on_val(AWV, CLS)
ARMS_A = [("uni", "no covariate"), ("raw", "raw past covariate"), ("randlag", "random lag (control)"), ("gcc_phat", "GCC-PHAT"), ("ccf", "CCF"),
          ("global", "global train lag"), ("ccf_pw", "CCF, prewhitened"), ("top3", "top-3 CCF peaks"), ("binned", "level-binned train lag"),
          ("lta_s0", "LTA, seed 0"), ("lta_s1", "LTA, seed 1"), ("lta_s2", "LTA, seed 2")]
tf_path = "results/air_timesfm/air_timesfm25_rows.csv"
def _tf_with_ctrl(main, ctrl):
    d = pd.read_csv(main); d = d[d.arm != "uni"]                      # D12: original uni arm read the backcast
    if os.path.exists(ctrl): d = pd.concat([d, pd.read_csv(ctrl)], ignore_index=True)   # corrected uni + randlag (D13)
    return AN.wide(d)
ATF = _tf_with_ctrl(tf_path, "results/air_timesfm_ctrl/air_timesfm25_rows.csv") if os.path.exists(tf_path) else None
L = [r"\begin{table}[t]\centering\small",
     rf"\caption{{AirLead test split (10 pairs, \num{{{len(AWT)}}} windows). Pooled GM-MASE ratios with 7-day block-bootstrap 95\% intervals and pairs improved (of 10). Best classical arm on validation (Chronos-2): {best_a.replace('_', chr(92)+'_')}. The random-lag row is the pre-registered specificity control.}}",
     r"\label{tab:air}", r"\begin{tabular}{lrrlrrlr}", r"\toprule",
     r" & \multicolumn{4}{c}{Chronos-2} & \multicolumn{3}{c}{TimesFM-2.5 (XReg)} \\",
     r"Arm & GM-MASE & ratio to raw & 95\% CI & pairs & GM-MASE & ratio to raw & pairs \\", r"\midrule"]
rows = []
for a, lab in ARMS_A:
    r1 = cmp_cells(AWT, a, "raw")
    if ATF is not None and a in ATF.columns:
        t1 = cmp_cells(ATF, a, "raw"); tg = f"{pooled(ATF, a):.3f}"
    else:
        t1 = ("--", "--", "--"); tg = "--"
    rows.append([a, pooled(AWT, a), *r1, tg, t1[0], t1[2]])
    L.append(f"{lab} & {pooled(AWT, a):.3f} & {r1[0]} & {r1[1]} & {r1[2]} & {tg} & {t1[0]} & {t1[2]} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("paper/tables/tab_air.tex", "w").write("\n".join(L))
pd.DataFrame(rows, columns=["arm", "gm", "r_raw", "ci", "pairs", "tf_gm", "tf_r_raw", "tf_pairs"]).to_csv("paper/tables/tab_air.csv", index=False)
hyp = {}
for a in ["lta_s0", "lta_s1", "lta_s2", "randlag"]:
    c = AN.compare(AWT, a, best_a); hyp[f"c2_{a}_vs_{best_a}"] = [c["pooled_ratio"], c["ci_lo"], c["ci_hi"], c["pairs_better"]]
c = AN.compare(AWT, best_a, "raw"); hyp[f"c2_{best_a}_vs_raw"] = [c["pooled_ratio"], c["ci_lo"], c["ci_hi"], c["pairs_better"]]
c = AN.compare(AWT, "randlag", "lta_s0"); hyp["c2_randlag_vs_lta_s0"] = [c["pooled_ratio"], c["ci_lo"], c["ci_hi"], c["pairs_better"]]
if ATF is not None:
    for a, r in [(best_a, "raw"), ("lta_s0", best_a), ("lta_s1", best_a), ("lta_s2", best_a)]:
        if a in ATF.columns and r in ATF.columns:
            c = AN.compare(ATF, a, r); hyp[f"tf_{a}_vs_{r}"] = [c["pooled_ratio"], c["ci_lo"], c["ci_hi"], c["pairs_better"]]
json.dump({"best_classical_val": best_a, "val_scores": va_scores, "tests": hyp}, open("paper/tables/air_hypotheses.json", "w"), indent=1)

# exploratory (post hoc) AirLead validation check
EX = pd.read_csv("results/air_exploratory_val.csv")
L = [r"\begin{table}[t]\centering\small", r"\caption{Post-hoc AirLead check (not pre-registered): Chronos-2, every 12th validation window (\num{" + str(len(EX)) + r"} windows). GM ratio to the raw covariate. All arms keep the raw past covariate and add one horizon-filled channel; ``noise'' is i.i.d.\ standard normal, known over context and horizon.}",
     r"\label{tab:air_explo}", r"\begin{tabular}{lrr}", r"\toprule", r"Added channel & ratio to raw & pairs improved \\", r"\midrule"]
for k, lab in [("fixed24", "fixed lag 24 h"), ("fixed48", "fixed lag 48 h"), ("fixed96", "fixed lag 96 h"), ("fixed168", "fixed lag 168 h"),
               ("ccf_h", "CCF lag restricted to $[24,168]$ h"), ("noise_known", "pure noise")]:
    pr = EX.groupby("cell").apply(lambda g: gmr(g[k], g["raw"]))
    L.append(f"{lab} & {gmr(EX[k], EX['raw']):.3f} & {(pr < 1).sum()}/{len(pr)} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("paper/tables/tab_air_explo.tex", "w").write("\n".join(L))
print("air tables written; best classical:", best_a, "timesfm included:", ATF is not None)

# ---------------- Deviations appendix (from protocol/DEVIATIONS.md) ----------------
import re
def tex_escape(t):
    t = t.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        t = t.replace(a, b)
    t = t.replace("->", r"$\rightarrow$").replace("≈", r"$\approx$").replace("≥", r"$\geq$").replace("≤", r"$\leq$").replace("×", r"$\times$")
    return re.sub(r"`([^`]*)`", r"\\texttt{\1}", t)
md = open("protocol/DEVIATIONS.md").read().split("\n## ")[1:]
L = [r"All deviations were logged when they occurred, in the time-stamped file \texttt{protocol/DEVIATIONS.md}; they are reproduced here verbatim except for formatting.", r"\begin{itemize}"]
for blk in md:
    head, *body = blk.strip().split("\n")
    L.append(r"\item \textbf{" + tex_escape(head.strip().rstrip(".")) + ".} " + tex_escape(" ".join(x.strip() for x in body if x.strip())))
L.append(r"\end{itemize}")
open("paper/tables/deviations.tex", "w").write("\n".join(L))
print("deviations:", len(md))

# ---------------- fit wide tables to text width ----------------
for fn in ["tab_gate", "tab_synth", "tab_ablation", "tab_river", "tab_air"]:
    p = f"paper/tables/{fn}.tex"; t = open(p).read()
    if r"\resizebox" not in t:
        t = re.sub(r"(\\begin\{tabular\}\{[^}]*\})", r"\\resizebox{\\linewidth}{!}{\1", t, count=1).replace(r"\end{tabular}", r"\end{tabular}}")
        open(p, "w").write(t)
print("resized")

# ---------------- Transfer (RiverLead test; aligned covariates from synthetic-only LTA seed 0) ----------------
import glob
def load_transfer():
    out = {}
    tf = "results/tr_timesfm/transfer_timesfm25_rows.csv"
    if os.path.exists(tf):
        d = pd.read_csv(tf)
        d = d[d.arm != "uni"]                                         # D12: original uni arm read the backcast
        ctl = "results/tr_timesfm_ctrl/transfer_timesfm25_rows.csv"   # D12/D13: corrected uni + random-lag control
        if os.path.exists(ctl):
            d = pd.concat([d, pd.read_csv(ctl)], ignore_index=True)
        out["TimesFM-2.5"] = d
    parts = sorted(glob.glob("results/tr_tabpfn_p*/transfer_tabpfn_ts_rows.csv"))
    full = "results/tr_tabpfn_full/transfer_tabpfn_ts_rows.csv"
    if os.path.exists(full):
        out["TabPFN-TS"] = pd.read_csv(full)
    elif parts:
        d = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(["arm", "cell", "id"])
        out["TabPFN-TS"] = d
    mo = "results/tr_moirai11_local/transfer_moirai11_rows.csv"
    if os.path.exists(mo):
        out["Moirai-1.1"] = pd.read_csv(mo)
    c2 = RT[RT.arm.isin(["uni", "raw", "top3", "binned", "ccf_pw", "randlag", "lta_s0"])].copy(); c2.loc[c2.arm == "lta_s0", "arm"] = "lta_zs"
    out = {"Chronos-2": c2, **out}
    return out
TR = load_transfer()
rows = []
for bb, d in TR.items():
    w = AN.wide(d); n = len(w)
    arms = [a for a in ["uni", "raw", "ccf_pw", "top3", "binned", "randlag", "lta_zs"] if a in w.columns and w[a].notna().all()]
    best = min([a for a in ["ccf_pw", "top3", "binned"] if a in arms], key=lambda a: AN.pooled_ratio(w, a, "raw"))
    r = {"backbone": bb, "n": n, "best_test_classical": best}
    for a in arms: r[f"gm_{a}"] = pooled(w, a)
    for a, ref in [("raw", "uni"), ("randlag", "raw"), (best, "raw"), ("lta_zs", "raw"), ("lta_zs", best), ("lta_zs", "top3"), ("lta_zs", "uni")]:
        if a in arms and ref in arms and a != ref:
            c = AN.compare(w, a, ref); r[f"{a}/{ref}"] = (c["pooled_ratio"], c["ci_lo"], c["ci_hi"], c["pairs_better"])
    rows.append(r)
TT = pd.DataFrame(rows); TT.to_csv("paper/tables/tab_transfer.csv", index=False)
def cell(r, key):
    v = r.get(key)
    return "--" if not isinstance(v, tuple) else f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}] {v[3]}/9"
def cell2(r, key):
    v = r.get(key)
    return ("--", "") if not isinstance(v, tuple) else (f"{v[0]:.3f} ({v[3]}/9)", f"[{v[1]:.3f}, {v[2]:.3f}]")
L = [r"\begin{table}[t]\centering\small",
     r"\caption{Transfer on the RiverLead test split. The same aligned covariates (synthetic-only \lta{}, seed 0, trained through Chronos-2) are fed to every backbone. Each cell: pooled GM-MASE ratio (pairs improved of 9) with block-bootstrap 95\% interval below. ``Best classical'' is chosen on each backbone's \emph{test} split among prewhitened CCF, top-3 peaks and the flow-binned lag, which favours the baseline. TabPFN-TS uses every 4th window (D5). The TimesFM-2.5 no-covariate arm is the corrected one (D12) and its random-lag control is post hoc (D13); random lag was not run for TabPFN-TS and Moirai-1.1.}",
     r"\label{tab:transfer}", r"\setlength{\tabcolsep}{4pt}\begin{tabular}{l" + "c" * len(rows) + "}", r"\toprule",
     "Ratio & " + " & ".join(r["backbone"] for r in rows) + r" \\",
     " & " + " & ".join(f"$n={r['n']}$" for r in rows) + r" \\", r"\midrule"]
SPECS = [("raw / no cov.", lambda r: "raw/uni"), ("random lag / raw", lambda r: "randlag/raw"),
         ("best class. / raw", lambda r: r["best_test_classical"] + "/raw"), ("\\lta{} / raw", lambda r: "lta_zs/raw"),
         ("\\lta{} / best class.", lambda r: "lta_zs/" + r["best_test_classical"]), ("\\lta{} / top-3", lambda r: "lta_zs/top3"),
         ("\\lta{} / no cov.", lambda r: "lta_zs/uni")]
for lab, kf in SPECS:
    cs = [cell2(r, kf(r)) for r in rows]
    L.append(lab + " & " + " & ".join(c[0] for c in cs) + r" \\")
    L.append(" & " + " & ".join(r"{\footnotesize " + c[1] + "}" for c in cs) + r" \\[2pt]")
L.append(r"\midrule")
L.append("best classical arm & " + " & ".join({"ccf_pw": "CCF-pw", "top3": "top-3", "binned": "binned"}[r["best_test_classical"]] for r in rows) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
t = "\n".join(L)
open("paper/tables/tab_transfer.tex", "w").write(t)
print("transfer table:", list(TR))

# ---------------- P5-A: synthetic transfer table (LTA vs plain CCF shift, by family, three backbones) ----------------
from scipy.stats import wilcoxon as _wx
def _syn_wide(paths_by_seed):
    out = {}
    for tag, (path, arm) in paths_by_seed.items():
        d = pd.read_csv(path); d = d[d.arm.isin([arm, "ccf", "raw", "oracle"])]
        w = d.pivot_table(index=["cell", "id"], columns="arm", values="mase")
        out[tag] = w
    return out
_b0 = pd.read_csv("results/main_k1/synth_test_rows.csv").pivot_table(index=["cell", "id"], columns="arm", values="mase")[["ccf", "raw", "oracle", "lta_zs"]]
SYN = {"Chronos-2": {"s0": _b0}}
for _s, _f in [("s1", "results/abl_seed1/synth_test_rows.csv"), ("s2", "results/abl_seed2/synth_test_rows.csv")]:
    _l = pd.read_csv(_f); _l = _l[_l.arm == "lta_zs"].set_index(["cell", "id"])["mase"]
    SYN["Chronos-2"][_s] = _b0[["ccf", "raw", "oracle"]].assign(lta_zs=_l.reindex(_b0.index).values)
    assert SYN["Chronos-2"][_s].lta_zs.notna().all()
for bb, f in [("TimesFM-2.5", "results/synthtr_timesfm/synthtr_timesfm25_rows.csv"), ("TabPFN-TS", "results/synthtr_tabpfn/synthtr_tabpfn_ts_rows.csv")]:
    d = pd.read_csv(f); seeds = sorted(a for a in d.arm.unique() if a.startswith("lta_s"))
    SYN[bb] = {s.replace("lta_", ""): d[d.arm.isin([s, "ccf", "raw", "oracle"])].pivot_table(index=["cell", "id"], columns="arm", values="mase").rename(columns={s: "lta_zs"}) for s in seeds}
FAMS = ["linear", "periodic", "nonlinear", "pulse", "tvlag", "null"]
FAML = {"linear": "fixed, linear", "periodic": "fixed, periodic", "nonlinear": "fixed, nonlinear", "pulse": "distributed", "tvlag": "time-varying", "null": "uninformative"}
srows = []
for bb, seeds in SYN.items():
    r = {"backbone": bb, "seeds": len(seeds)}
    rat = {}
    for s, w in seeds.items():
        lr = np.log(w["lta_zs"]) - np.log(w["ccf"]); r["n"] = len(lr)
        rat.setdefault("all", []).append(float(np.exp(lr.mean())))
        nz = lr[np.abs(lr) > 1e-9]; rat.setdefault("p", []).append(float(_wx(nz).pvalue))
        fm = lr.groupby(lr.index.get_level_values(0).map(lambda c: c.split("_")[0])).mean()
        for f_ in FAMS: rat.setdefault(f_, []).append(float(np.exp(fm[f_])))
    w0 = list(seeds.values())[0]
    r["oracle_raw"] = float(np.exp((np.log(w0["oracle"]) - np.log(w0["raw"])).mean()))
    r["ccf_raw"] = float(np.exp((np.log(w0["ccf"]) - np.log(w0["raw"])).mean()))
    for k, v in rat.items(): r[k] = (min(v), max(v))
    srows.append(r)
pd.DataFrame(srows).to_csv("paper/tables/tab_synth_transfer.csv", index=False)
def rng(t, d=3):
    a, b = t; return f"{a:.{d}f}" if abs(a - b) < 5 * 10 ** (-d - 1) else f"{a:.{d}f}--{b:.{d}f}"
def pfmt(t):
    hi = max(t); e = int(np.floor(np.log10(hi))) if hi > 0 else -300
    return r"$<10^{%d}$" % (e + 1) if hi < 1e-3 else f"{hi:.3f}"
L = [r"\begin{table}[t]\centering\small",
     r"\caption{Held-out synthetic test set on three frozen backbones. The aligned covariates come from the same three synthetic-only \lta{} seeds, trained through Chronos-2 only. Columns 3--4: GM-MASE ratios of the plain cross-correlation shift and of the oracle shift relative to the raw covariate (seed-independent). Remaining columns: GM of per-series MASE ratios \lta{}/cross-correlation shift, range over seeds (one seed for TabPFN-TS), overall and by family; $p$: largest Wilcoxon $p$ over seeds. TabPFN-TS uses 50 of the 100 series per cell.}",
     r"\label{tab:synth_transfer}", r"\begin{tabular}{lrrrllllllll}", r"\toprule",
     r" & & \multicolumn{2}{c}{vs raw} & \multicolumn{8}{c}{\lta{} / cross-correlation shift} \\ \cmidrule(lr){3-4}\cmidrule(lr){5-12}",
     r"Backbone & $n$ & CCF & oracle & all & $p$ & " + " & ".join(FAML[f_] for f_ in FAMS) + r" \\", r"\midrule"]
for r in srows:
    L.append(f"{r['backbone']} & {r['n']} & {r['ccf_raw']:.3f} & {r['oracle_raw']:.3f} & {rng(r['all'])} & {pfmt(r['p'])} & " + " & ".join(rng(r[f_]) for f_ in FAMS) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
t = "\n".join(L); t = re.sub(r"(\\begin\{tabular\}\{[^}]*\})", r"\\resizebox{\\linewidth}{!}{\1", t, count=1).replace(r"\end{tabular}", r"\end{tabular}}")
open("paper/tables/tab_synth_transfer.tex", "w").write(t)
print("synth transfer:", [(r["backbone"], rng(r["all"])) for r in srows])


# ---------------- P5-B: EnergyLead ----------------
ER = pd.read_csv("results/energy_chronos/energy_chronos2_rows.csv")
EWT, EWV = AN.wide(ER[ER.set == "energy_test"]), AN.wide(ER[ER.set == "energy_val"])
best_e, ve_scores = AN.best_on_val(EWV, CLS)
etf_path = "results/energy_timesfm/energy_timesfm25_rows.csv"
ETF = AN.wide(pd.read_csv(etf_path)) if os.path.exists(etf_path) else None
ELAGS = json.load(open("results/energy_chronos/energy_classical_lags.json"))
L = [r"\begin{table}[t]\centering\small",
     rf"\caption{{EnergyLead test split (10 pairs, \num{{{len(EWT)}}} windows; P5). Pooled GM-MASE ratios with 7-day block-bootstrap 95\% intervals and pairs improved (of 10). Best classical arm on validation (Chronos-2): {best_e.replace('_', chr(92)+'_')}. The fitted global train lags are 0--2\,h for nine of ten pairs (Ireland$\to$GB: 8\,h).}}",
     r"\label{tab:energy}", r"\begin{tabular}{lrrlrrlr}", r"\toprule",
     r" & \multicolumn{4}{c}{Chronos-2} & \multicolumn{3}{c}{TimesFM-2.5 (XReg)} \\",
     r"Arm & GM-MASE & ratio to raw & 95\% CI & pairs & GM-MASE & ratio to raw & pairs \\", r"\midrule"]
erows = []
for a, lab in ARMS_A:
    r1 = cmp_cells(EWT, a, "raw")
    if ETF is not None and a in ETF.columns:
        t1 = cmp_cells(ETF, a, "raw"); tg = f"{pooled(ETF, a):.3f}"
    else:
        t1 = ("--", "--", "--"); tg = "--"
    erows.append([a, pooled(EWT, a), *r1, tg, t1[0], t1[2]])
    L.append(f"{lab} & {pooled(EWT, a):.3f} & {r1[0]} & {r1[1]} & {r1[2]} & {tg} & {t1[0]} & {t1[2]} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
t = "\n".join(L); t = re.sub(r"(\\begin\{tabular\}\{[^}]*\})", r"\\resizebox{\\linewidth}{!}{\1", t, count=1).replace(r"\end{tabular}", r"\end{tabular}}")
open("paper/tables/tab_energy.tex", "w").write(t)
pd.DataFrame(erows, columns=["arm", "gm", "r_raw", "ci", "pairs", "tf_gm", "tf_r_raw", "tf_pairs"]).to_csv("paper/tables/tab_energy.csv", index=False)
ehyp = {}
for a, r in [(best_e, "raw"), ("lta_s0", best_e), ("lta_s1", best_e), ("lta_s2", best_e), ("randlag", "raw"), (best_e, "randlag"), ("lta_s0", "randlag"), ("raw", "uni"), ("lta_s0", "raw")]:
    c = AN.compare(EWT, a, r); ehyp[f"c2_{a}_vs_{r}"] = [c["pooled_ratio"], c["ci_lo"], c["ci_hi"], c["pairs_better"]]
if ETF is not None:
    for a, r in [(best_e, "raw"), ("lta_s0", best_e), ("lta_s1", best_e), ("lta_s2", best_e), ("randlag", "raw"), (best_e, "randlag"), ("raw", "uni"), ("lta_s0", "uni"), (best_e, "uni")]:
        if a in ETF.columns and r in ETF.columns:
            c = AN.compare(ETF, a, r); ehyp[f"tf_{a}_vs_{r}"] = [c["pooled_ratio"], c["ci_lo"], c["ci_hi"], c["pairs_better"]]
json.dump({"best_classical_val": best_e, "val_scores": ve_scores, "tests": ehyp, "lags": ELAGS}, open("paper/tables/energy_hypotheses.json", "w"), indent=1, default=str)
print("energy table: best", best_e, "timesfm", ETF is not None)

# ---------------- P6: TrafficLead ----------------
TRB = ["101N_401908", "87N_404452", "101N_402359", "680S_407184"]          # D25: imputed sensors
TRR = pd.read_csv("results/traffic_chronos/traffic_chronos2_rows.csv")
TWT, TWV = AN.wide(TRR[TRR.set == "traffic_test"]), AN.wide(TRR[TRR.set == "traffic_val"])
best_t, vt_scores = AN.best_on_val(TWV, CLS)
_cl = lambda w: w[~w.index.get_level_values(0).isin(TRB)]
ttf_path = "results/traffic_timesfm/traffic_timesfm25_rows.csv"
TTF = AN.wide(pd.read_csv(ttf_path)) if os.path.exists(ttf_path) else None
TLAGS = json.load(open("results/traffic_chronos/traffic_classical_lags.json"))
L = [r"\begin{table}[t]\centering\small",
     rf"\caption{{TrafficLead test split (PEMS-BAY, 5-min speeds; P6). Pooled GM-MASE ratios to the raw covariate with 7-day block-bootstrap 95\% intervals and pairs improved. `All': the ten pre-registered pairs (\num{{{len(TWT)}}} windows); `filtered': the six pairs without imputed sensors (D25). Best classical arm on Chronos-2 validation: {dict(ARMS_A).get(best_t, best_t)}. Train global lags: 3--29 steps for five pairs and 103--112 steps for the other five (D22).}}",
     r"\label{tab:traffic}", r"\begin{tabular}{lrrlrrrrr}", r"\toprule",
     r" & \multicolumn{6}{c}{Chronos-2} & \multicolumn{2}{c}{TimesFM-2.5 (XReg)} \\ \cmidrule(lr){2-7}\cmidrule(lr){8-9}",
     r"Arm & GM-MASE & ratio (all) & 95\% CI & pairs & ratio (filtered) & pairs & ratio (all) & ratio (filtered) \\", r"\midrule"]
trows = []
for a, lab in ARMS_A:
    r1 = cmp_cells(TWT, a, "raw"); r2 = cmp_cells(_cl(TWT), a, "raw")
    if TTF is not None and a in TTF.columns:
        t1 = cmp_cells(TTF, a, "raw")[0]; t2 = cmp_cells(_cl(TTF), a, "raw")[0]
    else:
        t1 = t2 = "--"
    trows.append([a, pooled(TWT, a), *r1, r2[0], r2[2], t1, t2])
    L.append(f"{lab} & {pooled(TWT, a):.3f} & {r1[0]} & {r1[1]} & {r1[2]} & {r2[0]} & {r2[2]} & {t1} & {t2} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
t = "\n".join(L); t = re.sub(r"(\\begin\{tabular\}\{[^}]*\})", r"\\resizebox{\\linewidth}{!}{\1", t, count=1).replace(r"\end{tabular}", r"\end{tabular}}")
open("paper/tables/tab_traffic.tex", "w").write(t)
pd.DataFrame(trows, columns=["arm", "gm", "r_raw", "ci", "pairs", "r_raw_f", "pairs_f", "tf_r_raw", "tf_r_raw_f"]).to_csv("paper/tables/tab_traffic.csv", index=False)
thyp = {}
for tag, W_ in [("all", TWT), ("filt", _cl(TWT))] + ([("tf_all", TTF), ("tf_filt", _cl(TTF))] if TTF is not None else []):
    for a, r in [(best_t, "raw"), ("lta_s0", best_t), ("lta_s1", best_t), ("lta_s2", best_t), ("randlag", "raw"), (best_t, "randlag"), ("lta_s0", "randlag"), ("raw", "uni"), ("lta_s0", "raw"), ("lta_s0", "uni"), (best_t, "uni")]:
        if a in W_.columns and r in W_.columns:
            c = AN.compare(W_, a, r); thyp[f"{tag}_{a}_vs_{r}"] = [c["pooled_ratio"], c["ci_lo"], c["ci_hi"], c["pairs_better"]]
json.dump({"best_classical_val": best_t, "val_scores": vt_scores, "tests": thyp, "lags": TLAGS, "excluded_D25": TRB}, open("paper/tables/traffic_hypotheses.json", "w"), indent=1, default=str)
print("traffic table: best", best_t, "timesfm", TTF is not None)

# ---------------- P5-C/D: cross-domain comparison (Chronos-2 frozen arms, CoRA, supervised baselines, fine-tuning) ----------------
def _n(d): d = d.copy(); d["id"] = d["id"].astype(str); return d[["arm", "cell", "id", "mase"]]
CR_ = pd.read_csv("results/cora/cora_rows.csv")
SUPP = {ds: (pd.read_csv(f"results/supervised/{ds}_rows.csv") if os.path.exists(f"results/supervised/{ds}_rows.csv") else None) for ds in ["river", "air", "energy", "traffic"]}
_tcr = pd.read_csv("results/traffic_cora/cora_rows.csv") if os.path.exists("results/traffic_cora/cora_rows.csv") else None
_rr = pd.read_csv("results/main_k1/river_rows.csv"); _rr = _rr[_rr.set == "river_test"]
DOMS = {"river": [_rr, pd.read_csv("results/ft_lora/ft_rows.csv"), pd.read_csv("results/ft_k1/ft_rows.csv"), CR_[CR_.set == "river_test"]],
        "air": [AR[AR.set == "air_test"], CR_[CR_.set == "air_test"]],
        "energy": [ER[ER.set == "energy_test"], CR_[CR_.set == "energy_test"]],
        "traffic": [TRR[TRR.set == "traffic_test"]] + ([_tcr] if _tcr is not None else [])}
BEST = {"river": "top3", "air": best_a, "energy": best_e, "traffic": best_t}
LTA0 = {"river": "lta_zs", "air": "lta_s0", "energy": "lta_s0", "traffic": "lta_s0"}
XW = {}
for ds, parts in DOMS.items():
    if SUPP[ds] is not None: parts = parts + [SUPP[ds]]
    if os.path.exists(f"results/supervised/{ds}_log_rows.csv"): parts = parts + [pd.read_csv(f"results/supervised/{ds}_log_rows.csv")]   # D21, post hoc
    XW[ds] = AN.wide(pd.concat([_n(x) for x in parts]))
    if ds == "traffic": XW[ds] = _cl(XW[ds])                                  # D25: filtered pairs
XROWS = [("uni", "Chronos-2, no covariate", "frozen, zero-shot"), ("raw", "Chronos-2, raw past covariate", "frozen, zero-shot"),
         ("BEST", "Chronos-2 + best classical alignment", "frozen, zero-shot"), ("LTA", "Chronos-2 + \\lta{} (seed 0)", "frozen, zero-shot"),
         ("cora_nocov", "CoRA adapter, covariate zeroed", "frozen, trained adapter"), ("cora_raw", "CoRA adapter, raw covariate", "frozen, trained adapter"),
         ("arx_short", "ARX, short history$^{\\ddagger}$", "supervised"), ("lgbm_short", "LightGBM, short history$^{\\ddagger}$", "supervised"),
         ("arx_full", "ARX, full history", "supervised"), ("lgbm_full", "LightGBM, full history", "supervised"),
         ("lgbm_full_nocov", "LightGBM, full history, no covariate", "supervised"),
         ("lgbm_full_log", "LightGBM, full history, log response$^{\\dagger}$", "supervised"),
         ("ft_lora_raw", "Chronos-2 LoRA fine-tuned, raw covariate", "fine-tuned"), ("ft_full_raw", "Chronos-2 fully fine-tuned, raw covariate", "fine-tuned")]
L = [r"\begin{table}[t]\centering\small",
     r"\caption{All methods on the four real test splits, GM-MASE ratio to frozen Chronos-2 without covariate (lower is better; 7-day block-bootstrap 95\% intervals; pairs improved of 9, 10, 10 and 6). TrafficLead uses the six pairs without imputed sensors (D25). Best classical alignment selected on validation: RiverLead top-3 CCF peaks, AirLead prewhitened CCF, EnergyLead global train lag, TrafficLead level-binned lag. CoRA, ARX, LightGBM and fine-tuning are trained on each benchmark's training split (P5, P6). $^\dagger$Post hoc (D21). $^\ddagger$Last 90 days of training data; 14 days for TrafficLead, whose training split is 90 days long. ``--'': not run.}",
     r"\label{tab:cross}", r"\begin{tabular}{llllllllll}", r"\toprule",
     r" & & \multicolumn{2}{c}{RiverLead} & \multicolumn{2}{c}{AirLead} & \multicolumn{2}{c}{EnergyLead} & \multicolumn{2}{c}{TrafficLead} \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
     r"Method & Setting & ratio [95\% CI] & pairs & ratio [95\% CI] & pairs & ratio [95\% CI] & pairs & ratio [95\% CI] & pairs \\", r"\midrule"]
xcsv = []
for key, lab, setting in XROWS:
    cells = []
    for ds in ["river", "air", "energy", "traffic"]:
        a = {"BEST": BEST[ds], "LTA": LTA0[ds]}.get(key, key); w = XW[ds]
        if a not in w.columns or w[a].isna().any():
            cells += ["--", "--"]; continue
        if a == "uni":
            cells += ["1", "--"]; xcsv.append((ds, key, 1.0, 1.0, 1.0, None)); continue
        c = AN.compare(w, a, "uni"); cells += [f"{c['pooled_ratio']:.3f} [{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]", str(c["pairs_better"])]
        xcsv.append((ds, key, c["pooled_ratio"], c["ci_lo"], c["ci_hi"], c["pairs_better"]))
    L.append(f"{lab} & {setting} & " + " & ".join(cells) + r" \\")
    if key in ("LTA", "cora_raw", "lgbm_full_log"): L.append(r"\addlinespace")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
t = "\n".join(L); t = re.sub(r"(\\begin\{tabular\}\{[^}]*\})", r"\\resizebox{\\linewidth}{!}{\1", t, count=1).replace(r"\end{tabular}", r"\end{tabular}}")
open("paper/tables/tab_cross.tex", "w").write(t)
pd.DataFrame(xcsv, columns=["domain", "method", "ratio_uni", "lo", "hi", "pairs"]).to_csv("paper/tables/tab_cross.csv", index=False)
print("cross table rows", len(xcsv), "supervised available", {k: v is not None for k, v in SUPP.items()})


# ---------------- D24: training-split lead diagnostic (post hoc, descriptive) ----------------
from lta import dm as DMM
DGT = pd.read_csv("results/lead_diagnostic_train.csv")
STEP = {"river": "1 h", "air": "1 h", "energy": "1 h", "traffic": "5 min"}
NAMES = {"river": "RiverLead", "air": "AirLead", "energy": "EnergyLead", "traffic": "TrafficLead"}
L = [r"\begin{table}[t]\centering\small",
     r"\caption{Training-split lead diagnostic (post hoc, D24). Global lag and mean cross-correlation (CCF) peak at that lag on log-differences, per pair, from training windows only; and the per-pair test ratio of the validation-selected classical arm to the raw covariate with Chronos-2 (TrafficLead: filtered pairs). Only RiverLead combines a multi-step lead with a non-trivial correlation at that lag, and only there did alignment give a lead-specific gain.}",
     r"\label{tab:leaddiag}", r"\begin{tabular}{lllll}", r"\toprule",
     r"Benchmark & step & train lag, median [range] & train CCF peak, median [range] & classical / raw, range over pairs \\", r"\midrule"]
for ds in ["river", "air", "energy", "traffic"]:
    g = DGT[DGT.domain == ds]
    if ds == "traffic": g = g[~g.pair.isin(TRB)]
    w = XW[ds]; rr_ = [v for k, v in AN.compare(w, BEST[ds], "raw", B=50).items() if k.startswith("r_")]
    L.append(f"{NAMES[ds]} & {STEP[ds]} & {g.train_lag.median():.0f} [{g.train_lag.min():.0f}, {g.train_lag.max():.0f}] & {g.train_ccf.median():.2f} [{g.train_ccf.min():.2f}, {g.train_ccf.max():.2f}] & {min(rr_):.3f}--{max(rr_):.3f} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
t = "\n".join(L); t = re.sub(r"(\\begin\{tabular\}\{[^}]*\})", r"\\resizebox{\\linewidth}{!}{\1", t, count=1).replace(r"\end{tabular}", r"\end{tabular}}")
open("paper/tables/tab_leaddiag.tex", "w").write(t)

# ---------------- Diebold-Mariano (HLN) per-pair tests ----------------
DMC = [("raw", "uni", "raw / no covariate"), ("BEST", "raw", "best classical / raw"), ("LTA", "raw", r"\lta{} / raw"),
       ("LTA", "BEST", r"\lta{} / best classical"), ("randlag", "raw", "random lag / raw"), ("cora_raw", "LTA", r"CoRA / \lta{}")]
L = [r"\begin{table}[t]\centering\small",
     r"\caption{Per-pair Diebold--Mariano tests with the Harvey--Leybourne--Newbold correction (Chronos-2 arms and CoRA; loss: log MASE per window, windows in time order, Newey--West variance). Entries: pairs significantly better / significantly worse at the 5\% level, of the number of pairs in brackets. TrafficLead: filtered pairs (D25).}",
     r"\label{tab:dm}", r"\begin{tabular}{lcccc}", r"\toprule",
     r"Comparison & RiverLead (9) & AirLead (10) & EnergyLead (10) & TrafficLead (6) \\", r"\midrule"]
dmcsv = []
for a0, b0, lab in DMC:
    cells = []
    for ds in ["river", "air", "energy", "traffic"]:
        a = {"BEST": BEST[ds], "LTA": LTA0[ds]}.get(a0, a0); b = {"BEST": BEST[ds], "LTA": LTA0[ds]}.get(b0, b0); w = XW[ds]
        if a not in w.columns or b not in w.columns or w[[a, b]].isna().any().any():
            cells.append("--"); continue
        _, bt, wr = DMM.dm_pairs(w, a, b); cells.append(f"{bt} / {wr}"); dmcsv.append((ds, lab, bt, wr))
    L.append(lab + " & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
t = "\n".join(L); t = re.sub(r"(\\begin\{tabular\}\{[^}]*\})", r"\\resizebox{\\linewidth}{!}{\1", t, count=1).replace(r"\end{tabular}", r"\end{tabular}}")
open("paper/tables/tab_dm.tex", "w").write(t)
pd.DataFrame(dmcsv, columns=["domain", "comparison", "sig_better", "sig_worse"]).to_csv("paper/tables/tab_dm.csv", index=False)
print("lead diag + DM tables written")
