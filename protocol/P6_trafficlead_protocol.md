# P6 — TrafficLead (fourth real domain; frozen 2026-11-22 before any pair screening or forecast)

Motivation: in RiverLead the lead is long and useful; in EnergyLead it turned out to be absent (<= 2 h for 9/10
pairs). TrafficLead tests a second domain where a multi-step lead is expected on physical grounds: congestion
waves travel UPSTREAM at roughly 15-20 km/h, so the speed at a downstream sensor 10-19 km away should lead the
speed at the upstream sensor by about 6-12 five-minute steps.

## Observed at freezing time
All results of P0-P5 and D1-D21. For PEMS-BAY only: array shape (52,116 x 325), time range
2017-01-01 00:00 to 2017-06-30 23:55 (5-min, local time, one DST gap on 2017-03-12), fraction of zero speeds
(3e-5), no NaN, and the sensor metadata (freeway, direction, absolute postmile, type). No speed series was
plotted, correlated or forecast.

## Data
PEMS-BAY (Li et al., 2018, DCRNN release), Caltrans PeMS District 4 loop detectors, average speed (mph) in
5-minute intervals. Obtained from the Kaggle mirror scchuy/pemsbay (pems-bay.h5, pems-bay-meta.h5); exported
inside a Kaggle kernel because the mirror's storage host is not reachable here. Sensor road distances from the
DCRNN repository were downloaded but are not used. Zero speeds are treated as missing; gaps <= 6 steps are
linearly interpolated; windows with remaining gaps or with a timestamp discontinuity are dropped.

## Pair selection (deterministic; metadata + TRAINING split only)
1. Candidates: mainline (Type ML) sensors on the same freeway and direction; the covariate is the DOWNSTREAM
   sensor, the target the UPSTREAM sensor (postmile increases with travel direction for N/E, decreases for S/W);
   separation 6-12 miles (9.7-19.3 km) along the postmile.
2. For each candidate, on training windows: global lag (lta.arms.fit_global_lag, same estimator as all other
   domains) and the mean CCF peak value at that lag on log-differences.
3. Keep candidates with global train lag in [3, 168] steps (>= 15 min).
4. Rank kept candidates by the mean CCF peak value (descending); greedily accept a pair if neither sensor is
   already used and its corridor (freeway, direction) has fewer than 2 accepted pairs; stop at 10 pairs.
   If fewer than 10 qualify, all qualifying pairs are used and the number is reported.
The test split plays no role in selection.

## Windows and splits
- Train 2017-01-01 to 2017-04-01, val 2017-04-01 to 2017-05-01, test 2017-05-01 to 2017-07-01.
- C = 512 steps (42.7 h), H = 24 steps (2 h), origins every 24 steps (2 h); time step 5 min.
- Event windows: range of the downstream (covariate) speed over the last 24 context steps >= 20 mph.
- MASE uses the same in-sample scale as all other benchmarks (lag-24 naive on the context); ratios per window
  are scale-invariant.

## Arms, backbones, hypotheses (as EnergyLead, P5-B)
- Chronos-2 on val + test: uni, raw, ccf, ccf_pw, gcc_phat, top3, global, binned (fitted on train), randlag,
  lta_s0, lta_s1, lta_s2. TimesFM-2.5 on test: the same arms (all classical arms, as D18).
- T1: best classical (selected on Chronos-2 val) beats raw (pooled CI excludes 1).
- T2: lta_s0 beats it (CI excludes 1, >= 6/10 pairs); T2-robust: all three seeds.
- T3 (specificity): the best aligned arm beats randlag (CI excludes 1); otherwise reported as non-specific.
- T4 (descriptive): every arm vs uni.
- The aligners are the frozen synthetic-only checkpoints; no retraining on traffic.

## Trained baselines (as P5-C/D)
- CoRA on frozen Chronos-2 (cora_raw, cora_nocov), trained on train windows, 2,000 steps, val checkpointing.
- ARX and LightGBM, per pair, identical features (lags in steps), with full train history and with the last
  14 days only (the train split is only 90 days long, so the P5 90-day condition would equal full history);
  LightGBM also without covariate. The log-response variant (D21) is run as well and labelled post hoc.

## Reporting
All outcomes are reported whatever they are. Deviations are logged in protocol/DEVIATIONS.md from D22.
