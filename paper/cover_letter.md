Dear Editors,

We submit the manuscript "Where leading covariates are lost in frozen time-series foundation models: horizon alignment versus adaptation in a pre-registered study of rivers, weather, wind power and traffic" for consideration in the International Journal of Forecasting.

Recent studies (SimpleTimeBench; Won, 2026) show that time-series foundation models (TSFMs) make poor use of covariates. Our paper builds on these diagnoses rather than repeating them. It asks a narrower, practical question: for a covariate that is observed only in the past and leads the target, where in the forecasting pipeline is its information lost, and what is the best way to recover it when the backbone must stay frozen?

The main contributions are:

1. A pre-registered mechanism test on three covariate-aware TSFMs (Chronos-2, TimesFM-2.5, TabPFN-TS). Moving a leading covariate into the forecast horizon lowers MASE by 31-39%, whereas context-only and random shifts do not, which locates the loss at the horizon.
2. A small learned aligner, trained once on synthetic data. It improves on classical delay estimation where lags vary with the system state or responses are distributed, and it transfers to other backbones without retraining.
3. Four new real-world benchmarks (rivers, air pressure, wind generation, freeway traffic) with frozen protocols and random-lag specificity controls. On them we compare zero-shot alignment with a trained frozen-backbone adapter (CoRA), supervised ARX and LightGBM models, and fine-tuning.

The results are mixed, and we report them as obtained. Zero-shot alignment is the most accurate frozen-backbone method on rivers, where the lead is long and strong. It does not help on wind generation and air pressure, where the relation is essentially simultaneous, or on traffic, where the lead is long but weak. A raw past-only covariate made three of four TSFMs worse than no covariate. All deviations from the protocols are listed in an appendix.

We believe the paper will interest forecasting practitioners who use foundation models with exogenous information, because it gives a checkable rule for when alignment, adaptation or no covariate is the better choice.

The manuscript has not been published and is not under consideration elsewhere. Code, benchmark definitions, protocols and per-window results will be released publicly.

Sincerely,
[Authors]
