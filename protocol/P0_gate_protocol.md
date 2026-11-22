# Phase 0 — Oracle-lag headroom gate (pre-registered)

Frozen 2026-11-21, before the full gate run. Any later change is listed under "Deviations" in the
results file, never edited here.

## Disclosure of what was seen before freezing
1. Kaggle API probe on a single toy series (y_t = 2 x_{t-24} + noise, C = 512, H = 24): MAE uni / raw /
   oracle-future-known = 0.88 / 0.31 / 0.068 (Chronos-2) and 0.80 / 0.75 / 0.062 (TimesFM-2.5).
2. Local CPU smoke test with `autogluon/chronos-2-small`, DEV split indices 0-3 (4 series x 10 cells):
   GM-MASE uni 0.685, raw 0.646, oracle 0.459, oracle_pastonly 0.660, ccf 0.510, ccf_pw 0.520, randlag 0.648.
   These DEV indices are re-used inside the gate sample (indices 0-49); the TEST split has never been generated.

## Data
- LeadBench-S generator `lta/synth.py` (seeded `SeedSequence((family, tau0, delta, split, index))`).
- C = 512, H = 24, seasonal period m = 24.
- 10 cells (family, tau0, delta): linear 24, linear 48, periodic 30, periodic 41, pulse 24, pulse 36,
  nonlinear 24, nonlinear 48, tvlag 48/12, tvlag 48/20. All have tau_t >= H, so the oracle-aligned
  covariate is fully known over the horizon.
- Gate sample: DEV split, indices 0-49 per cell (500 series). TabPFN-TS: indices 0-19 (200 series) for cost.

## Backbones (all frozen, default settings)
Chronos-2 (`amazon/chronos-2`, 120M), TimesFM-2.5 (`google/timesfm-2.5-200m-pytorch`, XReg covariates),
TabPFN-TS (local), Moirai-1.1-R-base. A backbone that fails to load is reported as such and excluded.

## Arms (one covariate channel each)
uni (no covariate); raw (x over context, future unknown); oracle (x_{t-tau_t}, context + known horizon);
oracle_pastonly (oracle over context only); ccf (shift by argmax |CCF| lag, k in 1..168, on context);
ccf_pw (same, AR(1)-prewhitened); randlag (shift by a seeded random lag in [H, 168]).
NaN handling per backbone is documented in `lta/backbones.py` (TimesFM: forward-fill / hold-last).

## Metrics
Primary: MASE of the median (point) forecast, scale = in-sample seasonal-naive MAE (m = 24).
Aggregation: geometric mean (GM) over series of per-series MASE ratios. Tests: two-sided Wilcoxon
signed-rank on per-series MASE differences.

## Pass rules
- **G1 (headroom).** For >= 2 backbones: GM[MASE(oracle) / MASE(ref)] <= 0.95 with Wilcoxon p < 0.01,
  where ref = the better of {uni, raw} by GM-MASE for that backbone (conservative).
- **G2 (specificity).** For the same backbones: GM[MASE(randlag) / MASE(oracle)] >= 1.05.
- Gate PASSES iff G1 and G2 hold. If it fails, the method track is stopped and the result is written up
  as a negative/diagnostic finding.

## Learnable-gap diagnostic (decides framing, not kill)
L_cell = GM[MASE(best of ccf, ccf_pw) / MASE(oracle)] per cell and backbone.
- Method framing ("learned alignment beats classical alignment") proceeds only if L >= 1.05 in >= 2
  families on >= 2 backbones.
- Otherwise the paper is framed as interface diagnosis + classical alignment, and the learned module is
  reported only where it adds value.
