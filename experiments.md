# Experiments

## Plan

| Model | Obs type | Run name | Status |
|-------|----------|----------|--------|
| MLP surrogate decoder (θ→waves) | all 25 params → 24 cont. waveforms | `exp-v1_mlp-surrogate_sims` | done |
| Flow (NPE) | Pas waveform + HR | `exp-v1_enc-pashr_maf5_sims` | done — best val NLL=+22.41 (early stop ep190/600) |
| Flow (NPE) | cath lab (4 waves + 5 scalars) | `exp-v1_enc-cathlab_maf5_sims` | done — best val NLL=−18.54 (early stop ep380/600) |
| Flow (NPE) | all 24 continuous sim waveforms | `exp-v1_enc-allwaves_maf5_sims` | done — best val NLL=−38.74 (early stop ep205/600) |

## Results

### `exp-v1_mlp-surrogate_sims` — MLP surrogate decoder

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

### `exp-v1_enc-pashr_maf5_sims` — Flow (Pas+HR)

600 epochs max, 1M sims, early stop at ep190. Best val NLL: **+22.41 nats**.

Pas+HR is a very low-information observation — single pressure waveform + one scalar. NLL staying
positive means the flow cannot substantially improve over the prior for most parameters. Expected:
most of the 25 parameters are unidentifiable from Pas alone.

### `exp-v1_enc-cathlab_maf5_sims` — Flow (cath lab)

600 epochs max, 1M sims, early stop at ep380. Best val NLL: **−18.54 nats**.

Full cath lab observation (Prv, Pra, Pvp, Pap + 5 scalars) gives substantial information — NLL
well below zero means the posterior is meaningfully tighter than the prior. Comparable to
cv-dann-sbi v3 sim NLL (12.84) modulo domain difference (pure sims, no DANN).

## Identifiability analysis — Pas+HR vs cath lab

Notebook: `notebooks/identifiability_cathlab_vs_pashr.ipynb`

Posteriors from both flows evaluated on 1000 held-out test sims.
Identifiability ratio = mean posterior std / prior std (1=uninformative, 0=fully identified).
NLL gap between the two flows: **~41 nats** (quantifies information value of right-heart pressures).

| Parameter | Pas+HR ratio | Cath lab ratio | Pas+HR R² | Cath lab R² | Note |
|-----------|-------------|----------------|-----------|-------------|------|
| Rap       | 0.583 | 0.006 | 0.543 | 1.000 | Right-heart only |
| Tmax      | 0.043 | 0.009 | 0.997 | 1.000 | Both |
| Ras       | 0.197 | 0.016 | 0.951 | 0.999 | Both (Pas+HR weaker) |
| Eap       | 0.714 | 0.017 | 0.409 | 0.998 | Right-heart only |
| Cas       | 0.096 | 0.020 | 0.987 | 0.999 | Both |
| Emax_RV   | 0.794 | 0.023 | 0.361 | 0.998 | Right-heart only |
| τ         | 0.725 | 0.024 | 0.381 | 0.995 | Right-heart only |
| Tmax_a    | 0.999 | 0.121 | 0.040 | 0.962 | Cath lab only |
| AVD       | 0.994 | 0.150 | 0.004 | 0.939 | Cath lab only |
| Brv       | 0.968 | 0.154 | 0.020 | 0.927 | Cath lab only |
| Eedref_rv | 0.835 | 0.177 | 0.338 | 0.833 | Cath lab only |
| Cvp       | 1.050 | 0.210 | 0.032 | 0.947 | Cath lab only |
| Vs        | 0.426 | 0.213 | 0.668 | 0.885 | Cath lab better |
| Emax_RA   | 1.010 | 0.241 | 0.032 | 0.926 | Cath lab only |
| Bra       | 1.002 | 0.273 | 0.013 | 0.844 | Cath lab only |
| Eedref_ra | 0.972 | 0.379 | 0.099 | 0.762 | Cath lab only |
| τ_a       | 0.954 | 0.413 | 0.034 | 0.693 | Cath lab only |
| Emax_LA   | 1.010 | 0.489 | 0.028 | 0.764 | Cath lab only |
| Emax_LV   | 0.115 | 0.490 | 0.984 | 0.738 | Pas+HR better |
| Blv       | 0.628 | 0.542 | 0.527 | 0.661 | Partial both |
| Eedref_lv | 0.563 | 0.666 | 0.626 | 0.491 | Partial both |
| Bla       | 0.987 | 0.679 | 0.035 | 0.529 | Poor both |
| Cvs       | 0.970 | 0.721 | 0.063 | 0.374 | Poor both |
| Eedref_la | 0.984 | 0.766 | 0.081 | 0.350 | Poor both |

**Summary:** cath lab well-identifies 14/24 params (ratio<0.25); Pas+HR only 4/24.
`Emax_LV` uniquely better identified from Pas+HR (full waveform) than cath lab (only 3 Pas scalars).
`Eedref_la` is the only parameter unidentifiable from both (ratio>0.75 for both).

**SV via surrogate** (posterior θ → S1 decoder → Vlv → SV):
- Pas+HR: R²=0.960, mean posterior std=4.6 ml
- Cath lab: R²=0.913, mean posterior std=8.4 ml

Pas+HR gives tighter SV because it observes the full Pas waveform (201 pts) while cath lab compresses
Pas to 3 scalars (mean/max/min), losing pulse-pressure shape information that directly encodes SV.

## `exp-v1_enc-allwaves_maf5_sims` — Flow (all 24 waveforms)

1M sims, early stop ep205/600. Best val NLL: **−38.74 nats**.

Note: initial run used a buggy `CVDataset` that returned 25-dim theta (including HR); re-run after
fix to return 24-dim `theta_infer` (HR excluded, consistent with other datasets). NLL improved from
−29.48 → −38.74, confirming the theta-dim mismatch was materially degrading the flow.

All 24 continuous sim waveforms (volumes, pressures, flows) provide the theoretical identifiability
ceiling — parameters structurally unidentifiable here cannot be recovered from any waveform-based
observation type in this model family. NLL gap vs cath lab: ~20 nats; vs Pas+HR: ~61 nats.

## Next

- Run 3-way identifiability notebook (`notebooks/identifiability_cathlab_vs_pashr.ipynb`) to compare
  posterior widths and R² across all three obs types using the corrected allwaves checkpoint.
