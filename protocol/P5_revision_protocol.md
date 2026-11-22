# P5 — Revision experiments for the IJF submission (frozen 2026-11-22, before any forecast below was run)

Motivation: an internal review raised (i) why LTA is needed if classical shifting works on Chronos-2, (ii) a single
real domain with lead-specific evidence, (iii) LTA-vs-classical on synthetic data only for Chronos-2, (iv) no
frozen-adapter baseline, (v) no simple supervised lagged baseline.

## Observed at freezing time
All results in P0-P4 and D1-D13, in particular: Chronos-2 RiverLead (classical ~8%, LTA 8.4-9.7% vs raw; LTA vs
best classical not robust across seeds), TimesFM-2.5 RiverLead (raw covariate 33% worse than no covariate; LTA
restores parity), TabPFN-TS / Moirai-1.1 (raw covariate harmful, alignment does not help), AirLead (non-specific
for Chronos-2). For the energy data: only the OPSD column list and per-year completeness (fraction non-missing)
were inspected. No energy forecasts, cross-correlations, lags or summary statistics of the energy series were
computed. CoRA: method section of Qin et al. (arXiv:2510.12681), Sec. 3, read; no CoRA run.

## A. Synthetic transfer (LeadBench-S test, same 13 cells and seeds as the Chronos-2 synthetic test)
- TimesFM-2.5 (XReg per series, corrected no-covariate path, D12): 100 series per cell (1,300). Arms: uni, raw,
  ccf, ccf_pw, top3, randlag, oracle, lta_s0, lta_s1, lta_s2.
- TabPFN-TS (v2): series 0-49 of each cell (650) because of cost. Arms: uni, raw, ccf, top3, oracle, lta_s0.
- Aligned covariates: the three frozen synthetic-only aligners (seeds 0-2), applied in float32. No retraining.
- S1: per backbone, LTA (each seed run) vs ccf, pooled GM ratio < 1 with Wilcoxon p < 0.01 over all series.
  Per-family ratios are reported for every arm. S2 (descriptive): oracle vs raw, as in the gate.

## B. EnergyLead (third real domain: measured wind generation, Open Power System Data, 2020-10-06 release)
- Hourly UTC actual wind generation (TSO data via ENTSO-E/OPSD). Ten upwind -> downwind pairs fixed from
  geography only (prevailing westerlies; adjacent bidding zones/countries; >= 98% complete 2015-2019):
  IE->GB_GBN, GB_GBN->NL(onshore), NL(onshore)->DE_tennet, FR->DE_amprion, DE_amprion->DE_50hertz, DK_1->DK_2,
  DE_50hertz->PL, DK_2->SE_4, PL->LT, SE_3->FI. Belgium excluded (90-94% complete).
- Splits: train 2015-01-01 to 2018-01-01, val 2018-01-01 to 2019-01-01, test 2019-01-01 to 2020-10-01.
  Daily origins at 00 UTC, C = 512, H = 24, gaps <= 6 h linearly interpolated, windows with remaining gaps dropped.
- Event windows: range of the upwind series over the last 24 context hours >= 50% of its train-split
  99th percentile (a large upwind ramp).
- Backbones and arms exactly as AirLead (P4): Chronos-2 on val + test with uni, raw, ccf, ccf_pw, gcc_phat, top3,
  global, binned (upwind-level quintiles, fitted on train), randlag, lta_s0, lta_s1, lta_s2. TimesFM-2.5 on
  test with uni, raw, randlag, the Chronos-2 val-selected classical arm, top3, binned, lta_s0, lta_s1, lta_s2.
- Hypotheses (both backbones unless stated): E1 best classical (selected on Chronos-2 val) beats raw (pooled CI
  excludes 1). E2 lta_s0 beats it (CI excludes 1, >= 6/10 pairs); E2-robust: all three seeds. E3 (specificity):
  the best aligned arm beats randlag (CI excludes 1). If E3 fails, the energy result is reported as non-specific,
  as for AirLead. E4 (descriptive): every arm vs uni.

## C. Supervised lagged baselines (RiverLead, AirLead, EnergyLead; CPU)
- Per pair, direct multi-horizon (one model per horizon step h = 1..24), trained on hourly origins in the train
  split, early stopping / regularisation chosen on the val split, evaluated on exactly the test windows used
  for the TSFMs (same MASE, scale from the window's own context, m = 24).
- Features at origin t (all from the context): target lags 1-48 and 72, 96, 120, 144, 168; covariate lags 1-168;
  hour of day (sin/cos); day of week. Target: y_{t+h} - y_t. Everything standardised with train statistics.
- ARX: ridge regression, alpha from {0.1, 1, 10, 100, 1000} on val.
- LightGBM: 31 leaves, learning rate 0.05, min_data_in_leaf 50, feature_fraction 0.8, up to 1,000 rounds with
  early stopping (50) on val.
- Two history conditions: full (whole train split) and short (last 90 days of the train split only).
- Also ARX/LightGBM without covariate features (target-only), to measure the covariate's value for them.
- Reported descriptively against uni, raw, best classical, LTA and fine-tuned Chronos-2 (RiverLead), with
  pooled ratios and block-bootstrap CIs. No pass rule; the frozen-model setting is not expected to win against
  full-history supervised models.

## D. CoRA frozen-adapter baseline (Chronos-2; RiverLead, AirLead, EnergyLead)
- Implemented from Qin et al. Sec. 3 on Chronos-2 (the paper used Sundial): covariate embedding = Chronos-2
  encoder state at the last context patch of the covariate run univariately; target embedding = encoder states
  at the future-patch tokens of the univariate target run (the input to the Chronos-2 output head); zero-initialised
  projection; Granger Causality Embedding (softmax gate; with one covariate it is the identity, stated as such);
  MLP -> (gamma, beta, alpha); forecast = (1 + alpha) * Head(gamma + (1 + beta) * E_target), alpha per horizon
  step in the normalised scale. Backbone frozen.
- Trained per benchmark on the train split (daily origins, all pairs pooled), quantile loss, AdamW lr 1e-3,
  batch 64, 2,000 steps, checkpoint by val GM-MASE every 250 steps. Arm name: cora_raw.
- Compared with uni, raw, best classical, LTA (zero-shot), and on RiverLead with fine-tuned Chronos-2.
- Question answered: does a trained frozen adapter that re-weights an unshifted covariate match zero-shot
  shifting? Reported descriptively with CIs; no pass rule.

## Deviations
Logged in protocol/DEVIATIONS.md as D14 onwards.
