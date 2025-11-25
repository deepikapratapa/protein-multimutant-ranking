## Model summary — 1EY0A RF surrogate

### Data
- Protein: **1EY0A** (Staphylococcal nuclease, chain A)
- N single mutants: **482**
- Target: experimental ΔΔG (kcal/mol)

### Features

**Sequence / biochemical (12 features):**
- `pos` (res_index)
- Δhydropathy, Δcharge, Δvolume, Δpolarity
- `blosum62`
- `is_gly`, `is_pro`, `is_to_gly`, `is_to_pro`
- `pH`, `temp_C`

**Structural (DSSP-based) features:**
- `ss_H`, `ss_E`, `ss_C` (one-hot secondary-structure class)
- `asa` (absolute solvent-accessible surface area)
- `asa_norm` (ASA normalized by max ASA for that residue)
- `is_buried`, `is_exposed`

Total numeric features used in the RF: **20**.

### Model

- Regressor: `RandomForestRegressor`
- Implementation: `sklearn.pipeline.Pipeline(StandardScaler + RF)`
- File: `results/models_structural/rf_structural_1EY0A.pkl`

### Performance (5-fold CV, 1EY0A)

| Model type                 | CV R² (mean) | CV RMSE (kcal/mol) |
|---------------------------|-------------:|--------------------:|
| RF, sequence-only         | **0.36**     | **1.08**            |
| RF, seq + structural DSSP | **0.58**     | **0.87**            |

Train performance (seq + structural):
- Train R² ≈ **0.90**
- Train RMSE ≈ **0.44 kcal/mol**
- Train MAE ≈ **0.31 kcal/mol**

### Usage in GA

This RF + structural model is used as the **ΔΔG surrogate scorer** in the GA phase. It is loaded via:

- `src/model_api.py`
- Model path: `results/models_structural/rf_structural_1EY0A.pkl`
- Feature order: `numeric_features` as defined in `03_ML_Model_1EY0A_structural.ipynb`