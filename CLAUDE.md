# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## What this project is

Pure-sim identifiability analysis for cardiovascular SBI. No real patients, no domain adaptation. Goals:
- Quantify what each observation type tells you about each parameter
- Build a surrogate forward model (θ → waveforms) for deriving waveform quantities from posterior samples
- Compare Flow A (Pas + HR) vs Flow B (cath lab) posterior variance to study identifiability

Forked from `cv-dann-sbi` (sim-to-real gap repo). DANN/WDGRL components removed.

## Approach

- **No domain adaptation**: sims only, no WDGRL, no Mixup reals
- **Inference**: NPE via `sbi` package (v0.26) with different observation types
- **Surrogate**: supervised encoder (obs → θ) + decoder (θ → waveforms), trained purely on sims

## Data layout

Same as cv-dann-sbi:
- HDF5 files under `<data-root>/train/` and `<data-root>/test/`
  - Adamant: `/media/local/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314`
- `manifest_train.json` / `manifest_test.json` at `<data-root>/`
- `norm_stats.json`: wave normalisation stats (not committed)
- 25 variable parameters defined by `pvar_low`/`pvar_high` in manifest config

## Key constants

| Symbol | Value | Meaning |
|---|---|---|
| `N_PARAMS` | 25 | All parameters (including HR) |
| `N_PARAMS_INFER` | 24 | Parameters inferred by flow (excl. HR) |
| `N_CHANNELS` | 28 | Total waveform channels (24 continuous + 4 valve) |
| `N_CONT` | 24 | Continuous waveform channels |
| `T` | 201 | Time steps per waveform |

## Observation types

| Name | Input | Use |
|---|---|---|
| `pas_hr` | Pas waveform (201pts) + HR scalar | Flow A — minimal cath data |
| `cath_lab` | Prv/Pra/Pvp/Pap (4×201) + 5 scalars | Flow B — full cath lab |

## Surrogate decoder

`SurrogateDecoder` in `models.py`: θ (25-dim) → 24 continuous waveforms × 201 pts.
Used to map posterior θ samples → predicted waveforms → derived quantities (e.g. SV from Vlv).

## Run naming convention

`{type}-{obs_type}-v{version}`

- `type`: `exp` or `dry`
- `obs_type`: `pas-hr`, `cath-lab`, `surrogate`
- `v{version}`: integer or decimal

Examples:
- `exp-cath-lab-v1`
- `exp-pas-hr-v1`
- `exp-surrogate-v1`

## Experiment tracking

- `outputs/{run_name}/` — full runs (gitignored)
- `dry-runs/{run_name}/` — dry runs (gitignored)

## Git conventions

- Never add `Co-Authored-By: Claude` or any AI authorship trailer to commit messages.
- Always commit before running a full experiment.
