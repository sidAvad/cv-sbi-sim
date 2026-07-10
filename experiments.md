# Experiments

## Plan

| # | Model | Obs type | Run name | Status |
|---|-------|----------|----------|--------|
| S1 | MLP surrogate decoder (θ→waves) | all 25 params → 24 cont. waveforms | `exp-v1_mlp-surrogate_sims` | running |
| A1 | Flow (NPE) | Pas waveform + HR | `exp-v1_enc-pashr_maf5_sims` | pending |
| B1 | Flow (NPE) | cath lab (4 waves + 5 scalars) | `exp-v1_enc-cathlab_maf5_sims` | pending |

## Analysis

Once S1, A1, B1 are trained:
- Compare Var(Emax_RV | cath_lab) vs Var(Emax_RV | Pas+HR) across test sims
- Compare Var(SV | cath_lab) vs Var(SV | Pas+HR) — SV derived via surrogate decoder from posterior θ samples
- Calibration check for both flows on held-out test sims
