# Experiments

## Plan

| # | Model | Obs type | Run name | Status |
|---|-------|----------|----------|--------|
| S1 | Surrogate encoder + decoder | cath lab (all 24 cont. waves) | `exp-surrogate-v1` | pending |
| A1 | Flow (NPE) | Pas waveform + HR | `exp-pas-hr-v1` | pending |
| B1 | Flow (NPE) | cath lab (4 waves + 5 scalars) | `exp-cath-lab-v1` | pending |

## Analysis

Once S1, A1, B1 are trained:
- Compare Var(Emax_RV | cath_lab) vs Var(Emax_RV | Pas+HR) across test sims
- Compare Var(SV | cath_lab) vs Var(SV | Pas+HR) — SV derived via surrogate decoder from posterior θ samples
- Calibration check for both flows on held-out test sims
