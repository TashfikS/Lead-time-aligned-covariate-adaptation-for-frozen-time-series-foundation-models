# Deviations from pre-registered protocols

## D1 (P0 gate) — TimesFM-2.5 XReg batching error, fixed and rerun
First gate run (kernel lta-gate-k1) passed all 500 series to one `forecast_with_covariates` call. The XReg
library (`timesfm/utils/xreg_lib.py`, `fit`) solves a single least-squares problem over the stacked rows of all
series in a call, i.e. covariate coefficients are shared across the batch. With independent series whose
effect signs are random, the shared coefficient is near zero: all covariate arms gave GM-MASE 0.6583-0.6590.
This is an implementation error in our wrapper, not a protocol choice. Fix: one call per series (per-series
in-context regression, the documented intent). TimesFM-2.5 gate rerun as kernel lta-gate-k3-timesfm; the
first run is retained in results/gate_k1 and reported as the "batch-shared XReg" variant.

## D2 (synthetic generator) — `null` family added after the gate
A `null` family (beta = 0, covariate uninformative) was added to `lta/synth.py` for LTA training and as a
do-no-harm test. It was not part of the gate cells; gate results are unaffected (seeds depend on family code).

## D3 (P2 rivers) — pipeline dry run touched 61 test windows
Before the full run, a CPU dry run of the main kernel (Chronos-2-small, LTA trained 3 steps, every ~150th
window per split: 61 train / 61 val / 61 test windows) and of the fine-tuning kernel (20 test windows) were executed to check the code path. No design
choice (arms, hyper-parameters, hypotheses) was changed after it. Its numbers are not reported.

## D4 (P2 rivers) — LTA river fine-tuning never improved validation; LTA-ft = LTA-zs
In all runs (main seed 0, seeds 1-2, ablations), every fine-tuning checkpoint (lr 1e-3, 150-step evaluation)
had worse val GM-MASE than the step-0 zero-shot aligner, so pre-registered early stopping returned the step-0
weights: LTA-ft and LTA-lopo are numerically identical to LTA-zs. Local diagnosis (12 CPU steps): the river
quantile loss is heavy-tailed (batch losses 0.15-1.07) and the aligner sharpens its temperature and shortens its
lead (known horizon fraction 0.70 -> 0.61). Reported as is; H2/H3 are evaluated with the identical weights.

## D5 (P2 transfer) — TabPFN-TS evaluated on every 4th test window
TabPFN-TS needs ~1.2 s per window on a T4; all arms on 9,046 windows would exceed the Kaggle session limit.
A fixed deterministic subsample (window index % 4 == 0; 2,262 windows, all pairs) is used for every TabPFN-TS arm.
Chronos-2, TimesFM-2.5 and Moirai-1.1 use all test windows.

## D6 (Chronos-2 LoRA baseline) — first attempt crashed (torchao/peft version clash); rerun with torchao removed.

## D7 (transfer) — float16 overflow in saved aligned covariates
The main kernel stored aligned covariates as float16, which overflows above 65,504 cfs (Mississippi, Missouri,
Ohio). The in-kernel Chronos-2 evaluation used float32 and is unaffected. For transfer, the aligned covariates
were regenerated in float32 on CPU from the saved seed-0 weights (lta_zs.pt); vs the GPU float16 copy: median
relative difference 1.9e-4, mask disagreement 0.014% of positions (threshold ties). To keep inputs identical
across backbones, Chronos-2 is re-evaluated on the regenerated arrays as well (arm `lta_zs`, transfer run).

## D8 (transfer) — Chronos-2 float32 re-evaluation and Moirai transfer run on local CPU
The Kaggle transfer kernels for Chronos-2 and Moirai-1.1 stayed queued (account GPU-session limits). Both were
run locally on CPU with tools/local_transfer.py: same test windows, same arm construction (lta/arms.py), same
float32 aligned arrays (D7), same model checkpoints (amazon/chronos-2; Salesforce/moirai-1.1-R-base). Only the
device differs; Moirai uses 100 samples and the median as in the GPU wrapper.

## D9 (AirLead) — post-hoc specificity check (not pre-registered)
Because the pre-registered random-lag control beat every aligned arm on AirLead test, an exploratory check was
run on every 12th VALIDATION window (305 windows, Chronos-2): fixed lags 24/48/96/168 h, CCF restricted to
[24,168] h, and an i.i.d. noise channel known over context and horizon. Reported as post hoc (appendix table of the post-hoc AirLead check);
it does not change any pre-registered outcome.

## D10 (statistics) — Wilcoxon tie tolerance
Paired synthetic MASE differences below 1e-6 (float32 rounding of identical backbone inputs) are set to exact
zeros before the Wilcoxon test. Without this, p-values for families with many ties differed across processes
(e.g. linear: 0.873 vs 0.899) because of last-bit differences in CSV round-trips. Conclusions are unchanged.

## D11 (transfer) — TabPFN-TS run split into four disjoint parts
The single TabPFN-TS transfer kernel stayed queued. The identical window set (every 4th test window, D5) was split
deterministically into four disjoint interleaved parts (idx[part::4]) run as separate kernels (two on T4, two on
P100). Arms, weights and inputs are unchanged; results are concatenated. Redundant queued kernels for Chronos-2
and Moirai (run locally, D8) were deleted unrun. The original single kernel eventually ran; it covers all
2,262 windows and reproduces the three completed parts exactly (8,480 overlapping rows, zero difference). The
full run is used; the remaining queued part was deleted unrun.

## D12 (TimesFM-2.5 wrapper) — no-covariate arm read the backcast instead of the forecast
TimesFM-2.5 was compiled with return_backcast=True (required for XReg). With this setting the library prepends the
backcast to the point forecast, and our no-covariate path sliced the FIRST H values (padded context) instead of the
LAST H. Only the TimesFM `uni` arm is affected (gate, RiverLead transfer, AirLead); all covariate arms go through
forecast_with_covariates, which slices the horizon correctly (verified in the library source). Found because the
river `uni` GM-MASE (2.91) was implausible. Fix: take the last H values; the `uni` arm was rerun with the same checkpoint on
the identical task sets: locally on CPU for the gate (tools/timesfm_uni_fix.py, which also recomputes the old slicing;
old GM-MASE 0.837 reproduced exactly, corrected 0.659), and on Kaggle GPU for RiverLead and AirLead test together
with the random-lag control (D13). Gate rule G1 for TimesFM is re-evaluated with the corrected reference.

## D13 (TimesFM-2.5) — post-hoc random-lag control on RiverLead and AirLead
The pre-registered TimesFM-2.5 arms did not include the random-lag control. Because the control changed the
interpretation of AirLead for Chronos-2, it was added post hoc for TimesFM-2.5 on the RiverLead and AirLead test
splits (same seeds and lag range [H, 168] as for Chronos-2), together with the corrected no-covariate arm (D12).
Recorded before these runs finished; reported as post hoc. Comparisons between LTA and classical arms are
unaffected (both add one aligned channel to the raw covariate; top-3 adds three).

## D14 (P5-A) — synthetic aligned covariates regenerated on CPU; pulse family numerically sensitive
For the synthetic transfer runs the three frozen aligners were applied on CPU in float32. Re-scoring a subset
with Chronos-2 on CPU reproduced the GPU per-series MASE to a median relative difference of 4e-7, except in the
sparse pulse family, where tiny numerical differences in the near-zero aligned values are amplified by the
backbone's scaling (25 pulse series: GM local/GPU 1.038, sd of log ratio 0.126; time-varying-lag series identical).
All transfer comparisons are within one backbone and use the same CPU arrays for every LTA arm.

## D15 (P5-A) — dry run touched 26 synthetic test series
A CPU dry run of the synthetic transfer kernel (TimesFM-2.5, 2 series per cell, 5 arms) checked the code path.
Nothing was changed after it and its numbers are not reported.

## D16 (P5-B) — binned classical lag: fallback when no window passes the correlation threshold
fit_binned_lag keeps only training windows whose CCF peak is >= 0.3. On EnergyLead some pairs have few such
windows (GB->NL: 4 of 132 in the subsample used by the estimator), and in the subsampled dry run a pair had none,
which crashed the estimator. Fallback added: if no window qualifies, the peaks of all windows are used. This case
never occurred on RiverLead or AirLead, so their results are unchanged. Found in the dry run (D17), before any
energy forecast was scored.

## D17 (P5-B) — EnergyLead dry run
A CPU dry run (Chronos-2-small, ~150 windows per split, val only, 4 arms) checked the code path. Its numbers are
not reported and nothing was changed after it except D16.

## D18 (P5-B) — TimesFM-2.5 EnergyLead runs every classical arm
To avoid waiting for the Chronos-2 validation selection, TimesFM-2.5 is run on the test split with all classical
arms (ccf, ccf_pw, gcc_phat, top3, global, binned) in addition to uni, raw, randlag and the three LTA seeds. The
hypothesis tests use only the arm selected on Chronos-2 validation, as pre-registered; other arms are descriptive.

## D19 (P5-D) — CoRA implementation details and a post-hoc no-covariate control
(a) The paper zero-initialises the projection and "the MLP". Zero-initialising every MLP layer gives zero
gradients for all parameters, so we follow the adaLN-Zero convention the paper cites (DiT): projection and the
MLP output layer zero-initialised, MLP first layer default-initialised. Verified: at initialisation the CoRA
median forecast equals frozen univariate Chronos-2 (max abs difference 0.031 on values ~6.8e4, float32).
(b) Because the adapter's bias terms can learn a covariate-independent recalibration of Chronos-2 on each
benchmark, a post-hoc control `cora_nocov` (same adapter and training, covariate embedding set to zero) is added.
It separates the effect of the covariate from that of in-domain adaptation.

## D20 (P5-D) — CoRA dry run
A CPU dry run (Chronos-2-small, ~60 windows per split, 6 steps) checked the code path on all three domains.
Its numbers are not reported and nothing was changed after it.

## D21 (P5-C) — post-hoc log-response LightGBM on RiverLead
The pre-registered supervised baselines model y_{t+h} - y_t in discharge units. Because discharge is heavy-tailed
and hydrological models usually work in log space, a post-hoc variant (`lgbm_full_log`) uses log(y) for all
target features, log(x) for covariate features and log y_{t+h} - log y_t as response (non-positive discharge,
35 hourly values at the Ohio downstream gauge, is floored at 0.1% of the pair's training median), with otherwise identical
settings (full history, validation early stopping). Added after the pre-registered RiverLead supervised results
were seen; reported as post hoc. The pre-registered arms are unchanged.


## D22 (P6) — TrafficLead pair screening outcome; descriptive short-lag subset defined before any forecast
The pre-registered selection (train split only) gave 10 pairs. Five have train global lags of 3-29 steps
(15 min to 2.4 h: 101S_400743 29, 85S_400528 4, 85S_400209 8, 85N_400336 3, 85N_400148 8); five have 103-112
steps (8.6-9.3 h: 680S_407184, 101N_402359, 680S_407185, 101N_401908, 87N_404452), which is implausible for
congestion propagation over 6-12 miles and likely reflects the daily commute pattern. Train CCF peaks are weak
(0.080-0.108). The primary analysis keeps all 10 pairs as pre-registered. A descriptive 'short-lag' subset (the
five pairs with train lag < 60 steps) is also reported. It was defined from training data only, before any
validation or test forecast, and is labelled post hoc.

## D23 (P6) — TrafficLead dry run
A CPU dry run of the evaluation kernel (Chronos-2-small, ~60 windows per split, val only, 6 arms) checked the
code path, as did a dry run of the CoRA kernel (6 steps, ~60 windows). Their numbers are not reported and
nothing was changed after them.

## D24 (post hoc, all domains) — training-split lead diagnostic
After the TrafficLead Chronos-2 result (no aligned arm beat the raw covariate although train lags were >= 3 steps),
the train global lag and the mean train CCF peak value at that lag (same estimator as the screening, log-
differences) were computed for every pair of all four real benchmarks and related to the per-pair test ratio of
the validation-selected classical arm (and LTA seed 0) to the raw covariate with Chronos-2. Descriptive only;
computed after the test results of RiverLead, AirLead and EnergyLead were known.

## D25 (P6) — TrafficLead: imputed sensors found; quality-filtered subset
The supervised baselines gave implausible LightGBM GM-MASE of 0.005-0.02 on two pairs. Inspection showed that
their target sensors (401908, 404452) repeat their value exactly one week earlier at 100% of validation and test
steps and ~92% of training steps, i.e. they carry imputed, not measured, speeds. Two further sensors exceed 50%
exact weekly repeats in one split (target 402359: 92% train; covariate 400665: 56% val); all other sensors are at
1-7%. The protocol did not anticipate imputed series. Found after the Chronos-2 test results were known. We
report both the pre-registered 10 pairs and a quality-filtered set of 6 pairs that excludes any pair with a sensor
above 50% exact weekly repeats in any split (excluded: 101N_401908, 87N_404452, 101N_402359, 680S_407184). The
Chronos-2 conclusions are the same on both sets. Supervised and cross-domain comparisons use the filtered set,
because a model can reproduce imputed values exactly.
