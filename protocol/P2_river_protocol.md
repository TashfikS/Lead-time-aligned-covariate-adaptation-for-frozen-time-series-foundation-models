# Phase 2 — RiverLead real-data benchmark (pre-registered)

Frozen 2026-11-21, before any forecast was run on river data. Later changes go to `DEVIATIONS.md`.

## Disclosure of what was seen before freezing
- Descriptive lead-lag analysis over the full 2015-10 to 2025-09 record (all splits): global CCF lag on
  hourly log-flow differences and windowed (512 h) CCF lags vs. upstream flow. Spearman(flow, lag) was
  negative in 7/9 pairs (-0.50 to -0.78). No forecasting model was run on river data before freezing.
- Synthetic gate (Chronos-2) result: classical CCF alignment ≈ oracle for static lags; gap only for
  time-varying lags.

## Data
USGS NWIS instantaneous discharge (parameter 00060), hourly means, 9 upstream -> downstream pairs:
delaware 01438500->01463500, susquehanna 01536500->01570500, potomac 01638500->01646500,
connecticut 01170500->01184000, james 02035000->02037500, missouri 06909000->06934500,
mississippi 07010000->07020500, ohio 03294500->03303280, allegheny_ohio 03049500->03086000.
Gaps <= 6 h linearly interpolated; windows touching longer gaps dropped.
Target = downstream discharge; covariate = upstream discharge (past-only: only x[:C] is ever used).
C = 512 h, H = 24 h, origins daily at 00:00 UTC.
Splits by time: train 2015-10-01..2021-09-30 (17,425 windows), val 2021-10-01..2022-09-30 (2,912),
test 2022-10-01..2025-09-30 (9,046). Event subset: upstream max/min >= 1.5 over the last 48 h of context
(25.4 % of test windows).

## Arms (frozen backbone in all cases; backbone input = [aligned channel(s)] + raw past covariate unless noted)
1. uni — no covariate. 2. raw — raw past-only covariate.
3. ccf — per-window argmax CCF lag (levels), 0..168. 4. ccf_pw — same, AR(1)-prewhitened.
5. gcc_phat — per-window GCC-PHAT lag. 6. global — one lag per pair from the train split (CCF on log-diffs).
7. binned — state-dependent classical lag: train-split windowed CCF lags (log-diffs) regressed on upstream
   flow quintile; at test time lag chosen by the quintile of mean upstream flow in the last 24 h.
8. randlag — seeded random lag in [1, 168] (specificity control).
9. LTA-zs — Lead-Time Aligner trained only on synthetic data (zero-shot on rivers).
10. LTA-ft — LTA fine-tuned on the river train split (all pairs pooled), early-stopped on val.
11. LTA-lopo — LTA fine-tuned with the evaluated pair held out (transfer to unseen rivers).
Reference adaptation baseline: Chronos-2 fine-tuned (official `fit`, raw covariate) on the train split.
Raw-covariate channel is kept in every aligned arm so alignment can only add information.

## Backbones
Chronos-2 (primary; LTA trained through it). TimesFM-2.5 (per-series XReg), TabPFN-TS, Moirai-1.1 receive the
aligned channels produced by the same frozen LTA (no re-training) — transfer test.

## Metrics and tests
Primary: MASE (seasonal-naive scale, m = 24, computed on the context). Secondary: scaled WQL over 9 quantiles
(Chronos-2 only), MAE on the event subset.
Aggregation: GM over windows within pair, then GM over pairs. Uncertainty: paired block bootstrap over
origins (block = 7 days, 2,000 resamples) for the GM ratio, per pair and pooled; Wilcoxon signed-rank on
per-pair GM ratios (n = 9).

## Hypotheses
- H1 (interface diagnosis): best classical alignment arm, selected on val per backbone, beats raw on the
  test split for Chronos-2 and >= 1 other backbone (pooled GM ratio < 1, bootstrap 95% CI excludes 1).
- H2 (learned alignment): LTA-ft beats the best classical arm (selected on val, including `binned`) on the
  test split for Chronos-2 (pooled CI excludes 1) and in >= 6/9 pairs. Reported separately on the event subset.
- H3 (transfer): LTA-lopo beats the best classical arm on the pooled test split.
- Any hypothesis not met is reported as not supported, with the numbers.
