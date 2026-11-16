# literature-priors-cad

Code, data and results for the paper *Published Diagnostic Statistics as Bayesian Priors for Coronary Artery Disease: Prior-Data Conflict and a Small-Sample Window* (Tejas Tovinkere Pattabhi, University of the Cumberlands).

The study builds Bayesian logistic-regression priors from published diagnostic statistics (the Diamond–Forrester pretest table, the test likelihood ratios of Diamond et al. 1980, and the 2012 CAD Consortium model). It diagnoses why face-value priors conflict with a referral cohort, attenuates them with a learned factor, and measures their value as the training sample shrinks. The data are the Cleveland Clinic cohort (UCI Heart Disease) and three other centers: Budapest, Long Beach VA and the Z-Alizadeh Sani dataset.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13. Exact versions are pinned in `pyproject.toml` and `uv.lock`.

```
uv sync
uv run pytest            # fast tests of the analysis logic (about 2 minutes)
```

## Reproduce the results

Run from `code/`, in this order. Times are for an 8-core 2020 Intel MacBook Pro. Every script writes to `results/`.

```
uv run python analysis.py             # main models, LOO, test set, cross-validation, learning curve (~40 min)
uv run python ppc_check.py            # posterior predictive check by chest-pain group (~1 min)
uv run python sensitivity.py          # univariable estimates, ridge, sensitivity variants (~10 min)
uv run python transfer.py auto        # M2-auto: design rule inside each learning-curve subset (~21 min)
uv run python transfer.py sites       # Budapest and Long Beach learning curves (~62 min)
uv run python transfer.py dca         # decision curves (full data), literature calculator
uv run python transfer.py extra       # recalibrated calculator, reweighted test set, fluoroscopy check
uv run python transfer.py beta11      # Beta(1, 1) prior on the attenuation factor at n = 25, 50
uv run python transfer.py wideint     # wide intercept prior at Budapest and Long Beach
uv run python extend.py cle           # CAD Consortium priors, M2-check, M2-h, decision curves at n = 25 (~46 min)
uv run python extend.py sites         # the same at the three other centers, including Z-Alizadeh Sani (~170 min)
uv run python extend.py tehw          # wide intercept prior at Z-Alizadeh Sani (~11 min)
uv run python robustness.py           # two-way bootstrap of SEs and decision-curve intervals
uv run python characteristics.py      # patient characteristics by center
uv run python make_figure.py          # Fig. 1
```

`SMOKE=1` runs a script with tiny settings to check that it works. The `extend.py` parts retry a sampling call if macOS kills a forked sampler process; the retry uses the same seed, so the results are identical.

## Data

| File | Source | License |
|---|---|---|
| `data/processed.{cleveland,hungarian,va,switzerland}.data`, `heart-disease.names` | A. Janosi, W. Steinbrunn, M. Pfisterer, R. Detrano, "Heart Disease," UCI Machine Learning Repository, 1989, doi:10.24432/C52P4X | CC BY 4.0 |
| `data/Z-Alizadeh sani dataset.xlsx`, `data/z-alizadeh-sani.csv` (the same sheet as CSV) | R. Alizadehsani, M. Roshanzamir, Z. Sani, "Z-Alizadeh Sani," UCI Machine Learning Repository, 2013, doi:10.24432/C5Q31T | CC BY 4.0 |

The Swiss file is included for completeness but not analyzed (93% prevalence).

## Reporting

`TRIPOD-AI-checklist.md` maps each TRIPOD+AI item (Collins et al., BMJ 2024) to the paper or explains why it does not apply.

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0, from the sources above.
