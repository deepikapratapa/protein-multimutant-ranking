# How to Build Mutation Feature Rows for GA

This document explains how to construct **per-mutation feature rows** that are
compatible with the trained Random Forest model
`results/models_structural/rf_structural_1EY0A.pkl`.

The model expects the following feature columns (see `feature_list.txt`):

- `pos`
- `delta_hydropathy`
- `delta_charge`
- `delta_volume`
- `delta_polarity`
- `blosum62`
- `is_gly`
- `is_pro`
- `is_to_gly`
- `is_to_pro`
- `ss_idx`
- `ss_H`
- `ss_E`
- `ss_C`
- `asa`
- `asa_norm`
- `is_buried`
- `is_exposed`

All GA individuals must be converted into one or more rows with **exactly these
columns** before calling `MutantScorer`.

---

## 1. Required inputs

For *each* single-site mutation you need:

- `pdb_id` = `"1EY0"`
- `chain`  = `"A"`
- `pos` (or `res_index`) = integer position (e.g. 12, 17, 58)
- `wt` = wild-type amino acid (single-letter)
- `mut` = mutated amino acid (single-letter)

You also need access to:

- The structural table for 1EY0A produced earlier:

  ```text
  data/processed/single_mut_1EY0A_struct.csv