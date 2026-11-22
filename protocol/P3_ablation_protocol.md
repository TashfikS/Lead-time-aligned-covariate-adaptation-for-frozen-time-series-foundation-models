# Phase 3 — LTA ablations and seed replicates (recorded 2026-11-21, before any LTA result was read)

All variants use the main-kernel pipeline (synthetic training -> synthetic TEST -> river val/test zero-shot ->
river fine-tune -> river val/test), identical hyper-parameters, Chronos-2 backbone, classical arms skipped.
- Seeds: full model with seed 0 (main run, also LOPO), seed 1, seed 2. Report mean and range over seeds.
- `static`  : dynamic offset head disabled (single soft lag per window).
- `noaux`   : oracle auxiliary losses (lag cross-entropy + aligned-covariate MSE) disabled during synthetic training.
- `noccf`   : CCF-residual initialisation disabled (ccf_init_scale = 0; lag logits from the CNN only).
Comparisons: each ablation vs the seed-0 full model and vs the best classical arm (selected on river val), on
(i) synthetic TEST per family and (ii) river TEST pooled + event subset, using P2 statistics.
The ablations are descriptive (no pass/fail rule); they are reported in full regardless of direction.
