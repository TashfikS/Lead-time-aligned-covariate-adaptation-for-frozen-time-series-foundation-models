# Phase 4 — AirLead second real-world benchmark (pre-registered)

Frozen 2026-11-21 after downloading the data and before any lag analysis or forecast on it. Pairs were chosen
a priori (tools/fetch_weather.py) along mid-latitude westerlies; none has been dropped or added.
Context: at freezing time, the RiverLead results were known (classical alignment beats raw ~8%; the learned
aligner beat the best classical arm for seed 0 only). This benchmark is the out-of-domain confirmation test.

## Data
ERA5 hourly mean-sea-level pressure (hPa) from the Open-Meteo archive API, 2015-10-01 .. 2025-09-30, 10 pairs
(upwind -> downwind): omaha_chicago, minneapolis_milwaukee, kansascity_stlouis, dallas_memphis, brest_paris,
dublin_london, amsterdam_berlin, paris_frankfurt, adelaide_melbourne, beijing_seoul.
Target = downwind pressure; covariate = upwind pressure (past only). Splits identical to RiverLead:
train 2015-10-01..2021-09-30, val 2021-10-01..2022-09-30, test 2022-10-01..2025-09-30.
Windows: C = 512 h, H = 24 h, daily origins at 00:00 UTC. Event subset: upwind pressure range over the last
24 context hours >= 8 hPa (rapid synoptic change).

## Arms (all aligned arms also receive the raw past covariate, as on RiverLead)
uni, raw, ccf, ccf_pw, gcc_phat, top3, global (train-split lag per pair), binned (train-split lag per upwind
pressure-level quintile), randlag, lta_s0 / lta_s1 / lta_s2 (the three synthetic-only aligners trained through
Chronos-2 in Phase 3; NO training or tuning on weather data).
Backbones: Chronos-2 (all arms); TimesFM-2.5 XReg per series (uni, raw, best classical arm from Chronos-2 val,
binned, lta_s0, lta_s1, lta_s2).

## Metrics and tests
Exactly as P2: MASE (m = 24), GM within pair then over pairs, 7-day block bootstrap (2,000) for the pooled
ratio, Wilcoxon over per-pair ratios (n = 10).

## Hypotheses
- W1: best classical arm (selected on val, Chronos-2) beats raw on test (pooled CI excludes 1), Chronos-2 and TimesFM-2.5.
- W2: lta_s0 beats the best classical arm on test for Chronos-2 (CI excludes 1, >= 6/10 pairs).
- W2-robust: W2 holds for all three seeds. The paper claims a learned-alignment benefit on AirLead only if W2-robust holds.
- W3: W2 holds on TimesFM-2.5 with the same (Chronos-trained) aligners.
