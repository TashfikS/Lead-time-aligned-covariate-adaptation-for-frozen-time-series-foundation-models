# Phase 0 gate — results (protocol: P0_gate_protocol.md)

DEV split, 50 series/cell (TabPFN-TS 20/cell), C=512, H=24. GM of per-series MASE ratios; Wilcoxon two-sided.
ref = better of {uni, raw}. G1: GM[oracle/ref] <= 0.95, p < 0.01. G2: GM[randlag/oracle] >= 1.05.

| backbone              | ref   |   G1_ratio |   G1_p |   G1 |   G2_ratio |   G2 |   L_overall |
|:----------------------|:------|-----------:|-------:|-----:|-----------:|-----:|------------:|
| chronos2              | raw   |     0.6939 | 0      |    1 |     1.4849 |    1 |      1.0884 |
| timesfm25             | raw   |     0.6753 | 0      |    1 |     1.473  |    1 |      1.0876 |
| tabpfn_ts             | raw   |     0.6125 | 0      |    1 |     1.6151 |    1 |      1.1344 |
| moirai11              | uni   |     1.0122 | 0.0688 |    0 |     1.0013 |    0 |      0.9792 |
| timesfm25_batchshared | raw   |     0.9996 | 0.3304 |    0 |     1.0007 |    0 |      0.9939 |

**Verdict: gate PASSES** (G1 and G2 on Chronos-2, TimesFM-2.5 [per-series XReg] and TabPFN-TS).
- Moirai-1.1 fails G1 because it does not consume covariates at all: toy y = 3z with z known over the horizon gives MAE 0.895 with vs 0.905 without the covariate (official GluonTS predictor, moirai-1.1-R-small). Used as a negative control.
- `timesfm25_batchshared`: first run with one XReg fit shared across the batch (deviation D1); reported for transparency.
- Mechanism: oracle_pastonly (aligned covariate over the context only) gives no gain on any backbone; the gain comes entirely from the aligned covariate becoming known over the horizon.

**Learnable gap (framing rule).** Families with L = GM[best classical / oracle] >= 1.05: {'chronos2': ['tvlag'], 'timesfm25': ['linear', 'tvlag'], 'tabpfn_ts': ['nonlinear', 'tvlag']}.
The literal rule (>= 2 families on >= 2 backbones) is met by TimesFM-2.5 and TabPFN-TS, but the TimesFM linear gap is 1.051 (at threshold) and the only family with a gap on all three backbones is time-varying lag (L = 1.48-1.92). Conservative framing adopted: interface diagnosis + alignment; the learned aligner is claimed only where it beats classical alignment.

## Per-cell

| backbone   | cell         |   oracle/ref |   ccf/oracle |   ccfpw/oracle |   bestcl/oracle(L) |   randlag/oracle |   pastonly/ref |   p(ccf vs oracle) |
|:-----------|:-------------|-------------:|-------------:|---------------:|-------------------:|-----------------:|---------------:|-------------------:|
| chronos2   | linear_24    |        0.837 |        1.006 |          1.006 |              1.006 |            1.781 |          1.582 |              0.135 |
| chronos2   | linear_48    |        0.626 |        1     |          1     |              1     |            1.473 |          1.023 |              0.553 |
| chronos2   | nonlinear_24 |        0.765 |        1.014 |          1.021 |              1.01  |            1.339 |          1.027 |              0.198 |
| chronos2   | nonlinear_48 |        0.722 |        1.008 |          1.004 |              1     |            1.294 |          0.996 |              0.166 |
| chronos2   | periodic_30  |        0.667 |        1.003 |          1.003 |              1.001 |            1.612 |          1.07  |              0.304 |
| chronos2   | periodic_41  |        0.638 |        1.021 |          1.012 |              1.012 |            1.589 |          1.048 |              0.006 |
| chronos2   | pulse_24     |        0.789 |        1.075 |          1.044 |              1.02  |            1.244 |          1.001 |              0     |
| chronos2   | pulse_36     |        0.864 |        1.051 |          1.022 |              1.009 |            1.137 |          1     |              0.002 |
| chronos2   | tvlag_48_d12 |        0.553 |        1.506 |          1.59  |              1.479 |            1.738 |          0.996 |              0     |
| chronos2   | tvlag_48_d20 |        0.559 |        1.519 |          1.688 |              1.488 |            1.821 |          1.007 |              0     |
| timesfm25  | linear_24    |        0.53  |        1.011 |          1.011 |              1.011 |            1.923 |          1.116 |              0.103 |
| timesfm25  | linear_48    |        0.567 |        1.051 |          1.051 |              1.051 |            1.755 |          1.107 |              0.02  |
| timesfm25  | nonlinear_24 |        0.82  |        1.002 |          1.017 |              0.998 |            1.24  |          1.052 |              0.738 |
| timesfm25  | nonlinear_48 |        0.831 |        0.994 |          0.981 |              0.979 |            1.191 |          1.126 |              0.051 |
| timesfm25  | periodic_30  |        0.624 |        1.011 |          1.009 |              1.009 |            1.689 |          1.716 |              0.496 |
| timesfm25  | periodic_41  |        0.58  |        1.03  |          1.021 |              1.021 |            1.645 |          1.401 |              0.082 |
| timesfm25  | pulse_24     |        1.001 |        0.966 |          0.963 |              0.928 |            0.997 |          1.001 |              0.005 |
| timesfm25  | pulse_36     |        1.001 |        0.964 |          0.976 |              0.933 |            0.999 |          1.002 |              0.005 |
| timesfm25  | tvlag_48_d12 |        0.52  |        1.607 |          1.709 |              1.53  |            1.864 |          1.043 |              0     |
| timesfm25  | tvlag_48_d20 |        0.511 |        1.71  |          1.889 |              1.635 |            1.869 |          1.028 |              0     |
| tabpfn_ts  | linear_24    |        0.562 |        0.967 |          0.967 |              0.967 |            1.812 |          1.039 |              0.027 |
| tabpfn_ts  | linear_48    |        0.593 |        1.021 |          1.021 |              1.021 |            1.54  |          1.103 |              0.409 |
| tabpfn_ts  | nonlinear_24 |        0.539 |        1.133 |          1.17  |              1.101 |            1.907 |          0.973 |              0.027 |
| tabpfn_ts  | nonlinear_48 |        0.532 |        1.098 |          1.096 |              1.095 |            1.922 |          1.061 |              0.312 |
| tabpfn_ts  | periodic_30  |        0.58  |        1.021 |          1.019 |              1.019 |            1.719 |          1.422 |              0.076 |
| tabpfn_ts  | periodic_41  |        0.606 |        0.998 |          0.994 |              0.994 |            1.702 |          1.208 |              0.729 |
| tabpfn_ts  | pulse_24     |        0.909 |        1.077 |          1.061 |              1.015 |            1.07  |          0.978 |              0.048 |
| tabpfn_ts  | pulse_36     |        0.934 |        1.058 |          1.03  |              1.004 |            1.03  |          1.015 |              0.007 |
| tabpfn_ts  | tvlag_48_d12 |        0.585 |        1.535 |          1.602 |              1.497 |            1.637 |          0.986 |              0     |
| tabpfn_ts  | tvlag_48_d20 |        0.445 |        1.951 |          2.136 |              1.917 |            2.237 |          1.033 |              0     |