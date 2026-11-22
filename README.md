# Where leading covariates are lost in frozen time-series foundation models

Code, benchmarks, pre-registered protocols and per-window results for the paper
*"Where leading covariates are lost in frozen time-series foundation models: horizon alignment versus
adaptation in a pre-registered study of rivers, weather, wind power and traffic"*.

## Contents

| Path | What it is |
|---|---|
| `lta/` | Python package: synthetic generator (`synth.py`), backbone wrappers for Chronos-2, TimesFM-2.5, TabPFN-TS, Moirai-1.1 (`backbones.py`), Lead-Time Aligner (`aligner.py`, `train.py`), classical alignment arms (`arms.py`, `baselines.py`), benchmark windowing (`river.py`), CoRA adapter (`cora.py`), ARX / LightGBM baselines (`supervised.py`), statistics (`analysis.py`, `dm.py`) |
| `kaggle/` | Scripts run as Kaggle GPU kernels (gate, main training, ablations, fine-tuning, transfer, AirLead / EnergyLead / TrafficLead evaluation, CoRA) |
| `tools/` | Kernel builder, data fetchers, local runners (supervised baselines, local transfer, TimesFM fix) |
| `protocol/` | Pre-registered protocols P0-P6 and the deviation log `DEVIATIONS.md` (D1-D25) |
| `data/` | Benchmark series and pair definitions: RiverLead (`river/`), AirLead (`weather/`), EnergyLead (`energy/`), TrafficLead (`traffic/`) |
| `results/` | Per-window MASE rows, summaries, fitted lags, aligner checkpoints (`*.pt`) for every run |
| `paper/` | Manuscript (`main.tex`, `references.bib`), generated tables (`tables/`), figures (`figs/`), `make_tables.py`, cover letter |

`local_only/` (model weights, build directories, raw downloads, third-party PDFs) is not part of the repository.

## Environment

Python 3.11. Versions used locally: torch 2.15.0, chronos-forecasting 2.3.2, timesfm 3.0.2 (TimesFM-2.5
checkpoint `google/timesfm-2.5-200m-pytorch`), tabpfn 10.2.0, tabpfn-time-series, gluonts 0.17.0,
uni2ts 2.0.0, peft 0.22.0, lightgbm 4.7.0, scikit-learn 1.9.0, numpy 2.4.6, pandas 2.3.3, scipy 1.17.1.
Kaggle kernels install `chronos-forecasting==2.3.2`, `timesfm[torch,xreg]` and `tabpfn-time-series` at run time.

## Reproducing

1. Regenerate every table from the stored per-window results: `python paper/make_tables.py`
2. Compile the manuscript: `cd paper && tectonic main.tex` (or `latexmk -pdf main.tex`; class `elsarticle`).
3. Re-run experiments: build a kernel with `python tools/build_kernel.py <script> <outdir> <slug> <owner> key=value ...`
   and push it with the Kaggle CLI (`tools/kg.sh` reads the API token from an environment variable; no token is
   stored in this repository). Supervised baselines run on CPU: `python tools/run_supervised.py river,air,energy,traffic`.

## Data sources

- RiverLead: USGS NWIS instantaneous discharge (public domain), fetched with `tools/fetch_rivers.py`.
- AirLead: ERA5 surface pressure via the Open-Meteo historical archive (CC BY 4.0), `tools/fetch_weather.py`.
- EnergyLead: Open Power System Data, time series package 2020-10-06 (CC BY 4.0), doi:10.25832/time_series/2020-10-06.
- TrafficLead: PEMS-BAY (Li et al., ICLR 2018; Caltrans PeMS). The speed series are not redistributed here;
  `data/traffic/` holds the pair definitions only. Download PEMS-BAY (`pems-bay.h5`), then run
  `python tools/build_traffic.py path/to/pems-bay.h5` to rebuild `data/traffic/hourly.csv` (byte-identical to the file used).

## Licence

Code is released under the MIT License (see `LICENSE`). Data files keep the licences of their sources listed above.
