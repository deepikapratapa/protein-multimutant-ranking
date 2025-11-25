# 1EY0A Single-Mutation Stability Model — Summary

## Data

- Source: ProTherm/ThermoMutDB subset for protein **1EY0A** (Staphylococcal nuclease).
- `n ≈ 480` single-point mutations with experimental ΔΔG (kcal/mol).
- Conditions: pH ≈ 7, temperature around 20 °C (kept as input features).

## Features

Model input features include:

- **Experimental conditions**
  - `pH`, `temp_C`

- **Sequence-derived mutation descriptors**
  - `delta_hydropathy` (Kyte–Doolittle)
  - `delta_charge` (formal charge change at pH ~7)
  - `delta_volume` (side-chain volume difference)
  - `delta_polarity` (Grantham polarity difference)
  - `blosum62` (BLOSUM62 substitution score)

- **Mutation-type flags**
  - `is_gly`, `is_pro`, `is_to_gly`, `is_to_pro`

- **Structural context from DSSP**
  - `ss_idx` (encoded secondary structure: H/helix, E/β-strand, C/coil)
  - `ss_H`, `ss_E`, `ss_C` (one-hot secondary structure indicators)
  - `asa` (absolute solvent-accessible surface area)
  - `asa_norm` (normalized ASA)
  - `is_buried`, `is_exposed` (binary burial indicators)

## Model

- Regressor: **RandomForestRegressor** (`n_estimators=600`, `min_samples_leaf=2`, `random_state=42`)
- Preprocessing: StandardScaler on all numeric features.
- Evaluation: 5-fold cross-validation (randomized splits within 1EY0A).

## Performance (example)

- CV R² (mean ± std): ~0.25–0.30 (protein-specific; moderate signal)
- CV RMSE: ~1.2–1.3 kcal/mol
- Train R²: typically higher than CV, indicating some overfitting but acceptable for use as a **surrogate fitness function** in GA/MC search.

## Role in the Project

This model is used as a **protein-specific surrogate ΔΔG predictor** for 1EY0A:

- GA/MC will propose **multi-mutant candidates** on 1EY0A.
- These are decomposed into **single-mutation contributions** using the trained model.
- Approximate ΔΔG for multi-mutants is used to:
  - **rank** candidate variants,
  - **compare** GA vs Monte Carlo exploration strategies,
  - and select top candidates for downstream validation (e.g., FoldX).


