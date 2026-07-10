# Experiments

## Plan

| # | Model | Obs type | Run name | Status |
|---|-------|----------|----------|--------|
| S1 | MLP surrogate decoder (θ→waves) | all 25 params → 24 cont. waveforms | `exp-v1_mlp-surrogate_sims` | done |
| A1 | Flow (NPE) | Pas waveform + HR | `exp-v1_enc-pashr_maf5_sims` | running |
| B1 | Flow (NPE) | cath lab (4 waves + 5 scalars) | `exp-v1_enc-cathlab_maf5_sims` | running |

## Results

### S1 — MLP surrogate decoder (`exp-v1_mlp-surrogate_sims`)

300 epochs, 1M sims (998,976 train / 1,024 val), hidden=512 × 4 layers, lr=1e-3 + ReduceLROnPlateau.

Best val MSE: **0.0255**

Val R² per channel:

| Group | Channel | R² |
|-------|---------|-----|
| Pressure | Pap | 0.993 |
| | Pas | 0.994 |
| | Pla | 0.994 |
| | Plv | 0.987 |
| | Pra | 0.994 |
| | Prv | 0.991 |
| | Pvp | 0.996 |
| | Pvs | 1.000 |
| Flow | Qap | 0.978 |
| | Qas | 0.934 |
| | Qla | 0.900 |
| | Qlv | 0.876 |
| | Qra | 0.928 |
| | Qrv | 0.903 |
| | Qvp | 0.964 |
| | Qvs | 0.993 |
| Volume | Vap | 0.999 |
| | Vas | 0.999 |
| | Vla | 0.993 |
| | **Vlv** | **0.996** |
| | Vra | 0.995 |
| | Vrv | 0.997 |
| | Vvp | 0.998 |
| | Vvs | 1.000 |

**Mean R²: ~0.975.** P and V channels all >0.987. Q (flow) channels are hardest — Qlv=0.876, Qla=0.900 — due to sharp systolic peaks. Vlv R²=0.996 is excellent for SV derivation (SV = EDV − ESV).

### A1 — Flow (Pas+HR) (`exp-v1_enc-pashr_maf5_sims`)

600 epochs, 1M sims. Running.

### B1 — Flow (cath lab) (`exp-v1_enc-cathlab_maf5_sims`)

600 epochs, 1M sims. Running.

## Analysis

Once A1 and B1 are trained:
- Compare Var(Emax_RV | cath_lab) vs Var(Emax_RV | Pas+HR) across test sims
- Compare Var(SV | cath_lab) vs Var(SV | Pas+HR) — SV derived via S1 surrogate from posterior θ samples
- Calibration check for both flows on held-out test sims
