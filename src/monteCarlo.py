#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo (random search) baseline for multi-mutant design
using the same ML surrogate as the GA.

- Uses biochemical + structural features (whatever numeric features
  the model was trained on).
- Multi-mutant features are composed by SUMMING per-mutation feature vectors.
- Objective: MINIMIZE predicted ΔΔG (kcal/mol) → more negative = more stabilizing.

Inputs:
  - Trained surrogate model: results/models/best_model.pkl
  - Single-mutant feature table: data/processed/single_mut_clean.csv

Outputs (under results/ga_mc/):
  - mc_baseline.png : best-so-far ΔΔG vs iterations
  - top_variants.csv : MC best variant appended (method = "MC")

Run example (from repo root):
  python src/montecarlo_baseline.py \
      --model_path results/models/best_model.pkl \
      --singles_csv data/processed/single_mut_clean.csv \
      --mut_size 2 \
      --iters 7000 \
      --seed 123
"""

import os
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Helpers (mirrored from GA so behaviour is consistent)
# -------------------------------------------------------------------

NON_FEATURE_COLS_CANDIDATES = {
    "protein_id", "protein", "uniprot",
    "pos", "position", "site",
    "wt", "wild", "wt_res", "ref", "from",
    "mut", "mutant", "aa", "to",
    "chain", "temp",
    "ddg", "ΔΔG", "ddg_exp", "target"
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def infer_protein_col(df: pd.DataFrame) -> str:
    for c in ["protein_id", "protein", "uniprot", "ProteinID"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            return c
    return df.columns[0]


def infer_position_col(df: pd.DataFrame) -> str:
    for c in ["position", "pos", "Position", "site"]:
        if c in df.columns:
            return c
    int_cols = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
    return int_cols[0] if int_cols else df.columns[0]


def infer_wt_mut_cols(df: pd.DataFrame):
    wt_candidates = ["wt", "wild", "wt_res", "ref", "from"]
    mut_candidates = ["mut", "mutant", "aa", "to"]
    wt_col = next((c for c in wt_candidates if c in df.columns), None)
    mut_col = next((c for c in mut_candidates if c in df.columns), None)
    return wt_col, mut_col


def build_key(protein_id, pos, wt, mut) -> str:
    """
    Canonical mutation key: PROT:posWT>MUT
    Example: P12345:42A>W
    """
    if wt is None:
        wt = "?"
    if mut is None:
        mut = "?"
    return f"{protein_id}:{pos}{wt}>{mut}"


def select_feature_cols(df: pd.DataFrame) -> list:
    """
    Fallback: select numeric columns as features if model doesn't
    expose feature_names_in_. This naturally captures biochemical
    + structural numeric features.
    """

    numeric_cols = [
        # environment
        "pH", "temp_C",
        # sequence deltas
        "delta_hydropathy", "delta_charge",
        "delta_volume", "delta_polarity",
        "blosum62",
        # mutation flags
        "is_gly", "is_pro", "is_to_gly", "is_to_pro",
        # structural features from DSSP
        "ss_idx", "ss_H", "ss_E", "ss_C",
        "asa", "asa_norm", "is_buried", "is_exposed"]
    # numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feat_cols = [c for c in numeric_cols if c not in NON_FEATURE_COLS_CANDIDATES]
    return feat_cols or numeric_cols


def load_model(model_path: str):
    """
    Load best_model.pkl. If it has feature_names_in_, we enforce that
    same feature set/order.
    """
    model = joblib.load(model_path)
    if hasattr(model, "feature_names_in_"):
        return model, list(model.feature_names_in_)
    return model, None


def make_lookup(
    df: pd.DataFrame,
    protein_col: str,
    pos_col: str,
    wt_col: str,
    mut_col: str,
    feat_cols: list,
):
    """
    Build lookup structures:
      - key_to_feat: mutation key -> feature vector (np.array of len = n_features)
      - protein_to_positions: protein_id -> sorted list of integer positions
      - protein_pos_to_keys: (protein_id, pos) -> list of mutation keys at that site
      - key_meta: mutation key -> metadata dict (protein_id, pos, wt, mut)
    """
    key_to_feat = {}
    protein_to_positions = defaultdict(set)
    protein_pos_to_keys = defaultdict(list)
    key_meta = {}

    for _, row in df.iterrows():
        protein_id = row[protein_col]
        pos = int(row[pos_col])

        wt = row[wt_col] if wt_col in df.columns else None
        mut = row[mut_col] if mut_col in df.columns else None

        key = build_key(protein_id, pos, wt, mut)
        feat_vec = row[feat_cols].astype(float).to_numpy()

        key_to_feat[key] = feat_vec
        key_meta[key] = {
            "protein_id": protein_id,
            "pos": pos,
            "wt": wt,
            "mut": mut,
        }
        protein_to_positions[protein_id].add(pos)
        protein_pos_to_keys[(protein_id, pos)].append(key)

    protein_to_positions = {
        p: sorted(list(pos_set)) for p, pos_set in protein_to_positions.items()
    }
    return key_to_feat, protein_to_positions, protein_pos_to_keys, key_meta


def compose_features(keys, key_to_feat):
    """
    Multi-mutant features = SUM of per-mutation feature vectors.
    Includes biochemical + structural features.
    """
    feat_list = [key_to_feat[k] for k in keys]
    return np.sum(feat_list, axis=0)


def sample_valid_combo(
    mut_size: int,
    eligible_proteins: list,
    protein_to_positions: dict,
    protein_pos_to_keys: dict,
    rng: random.Random,
):
    """
    Sample one random valid multi-mutant:
      - pick a protein with >= mut_size positions
      - pick mut_size DISTINCT positions
      - pick one mutant key at each position
    """
    protein_id = rng.choice(eligible_proteins)
    positions = protein_to_positions[protein_id]
    if len(positions) < mut_size:
        return None

    chosen_positions = rng.sample(positions, mut_size)
    keys = []
    for pos in chosen_positions:
        cand = protein_pos_to_keys[(protein_id, pos)]
        keys.append(rng.choice(cand))

    # Sorted to have a canonical representation
    return tuple(sorted(keys))


# -------------------------------------------------------------------
# Monte Carlo core
# -------------------------------------------------------------------

def run_mc(args):
    outdir = os.path.join("results", "ga_mc")
    ensure_dir(outdir)

    # --- Load data ---
    singles = pd.read_csv(args.singles_csv)

    protein_col = infer_protein_col(singles)
    pos_col = infer_position_col(singles)
    wt_col, mut_col = infer_wt_mut_cols(singles)

    # --- Load model + enforce feature set ---
    model, model_feats = load_model(args.model_path)

    if model_feats is not None:
        missing = set(model_feats) - set(singles.columns)
        if missing:
            raise ValueError(
                "Your singles CSV is missing features used during training.\n"
                f"Missing columns: {sorted(missing)}\n"
                "Either (1) point MC to the same processed feature file used for training, "
                "or (2) update single_mut_clean.csv to include these columns."
            )
        feat_cols = list(model_feats)
    else:
        feat_cols = select_feature_cols(singles)

    print(f"[MC] Using {len(feat_cols)} features: {feat_cols}")

    key_to_feat, protein_to_positions, protein_pos_to_keys, key_meta = make_lookup(
        singles, protein_col, pos_col, wt_col, mut_col, feat_cols
    )

    # Proteins that can accommodate mut_size distinct positions
    eligible_proteins = [
        p for p, pos_list in protein_to_positions.items() if len(pos_list) >= args.mut_size
    ]
    if not eligible_proteins:
        raise ValueError(
            f"No proteins found with >= {args.mut_size} mutable positions. "
            f"Check your singles CSV and mut_size."
        )

    # --- Reproducibility ---
    rng = random.Random(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    best_curve = []
    best_ddg = float("inf")
    best_combo = None

    start = time.perf_counter()

    for i in range(1, args.iters + 1):
        combo = sample_valid_combo(
            args.mut_size,
            eligible_proteins,
            protein_to_positions,
            protein_pos_to_keys,
            rng,
        )
        if combo is None:
            continue

        x_multi = compose_features(combo, key_to_feat).reshape(1, -1)
        pred_ddg = float(model.predict(x_multi)[0])

        if pred_ddg < best_ddg:
            best_ddg = pred_ddg
            best_combo = combo

        best_curve.append(best_ddg)

        if i % max(1, args.iters // 10) == 0:
            print(f"[Iter {i:>5}] best ΔΔG = {best_ddg:.3f} kcal/mol")

    elapsed = time.perf_counter() - start
    print(
        f"MC finished {args.iters} iterations in {elapsed:.2f}s. "
        f"Best predicted ΔΔG = {best_ddg:.3f} kcal/mol"
    )

    # --- Plot convergence ---
    plt.figure(figsize=(6, 4))
    plt.plot(best_curve, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best-so-far predicted ΔΔG (kcal/mol) ↓")
    plt.title(f"Monte Carlo Baseline (mut_size = {args.mut_size}, iters = {args.iters})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    mc_plot_path = os.path.join(outdir, f"mc_baseline_{args.mut_size}.png")
    plt.savefig(mc_plot_path, dpi=200)
    plt.close()

    # --- Append MC best variant to top_variants.csv ---
    if args.out_variant_file:
        top_path = args.out_variant_file
    else:
        top_path = os.path.join(outdir, f"mc_top_variants_{args.mut_size}.csv")

    rows = []
    if best_combo is not None:
        protein_id = next(iter({key_meta[k]["protein_id"] for k in best_combo}))
        pos_list = [key_meta[k]["pos"] for k in best_combo]
        muts = [f"{key_meta[k]['wt']}>{key_meta[k]['mut']}" for k in best_combo]
        pretty = [f"{p}{m}" for p, m in sorted(zip(pos_list, muts))]

        rows.append({
            "protein_id": protein_id,
            "mutations": ";".join(best_combo),
            "pretty_sites": ";".join(map(str, pretty)),
            "num_mutations": len(best_combo),
            "predicted_ddg": best_ddg,
            "method": "MC",
        })

    df_mc = pd.DataFrame(rows)

    if os.path.exists(top_path):
        base_df = pd.read_csv(top_path)
        # If older GA file didn't have 'method', assume GA
        if "method" not in base_df.columns:
            base_df["method"] = "GA"
        out_df = pd.concat([base_df, df_mc], ignore_index=True)
    else:
        out_df = df_mc

    out_df.to_csv(top_path, index=False)

    print(f"Saved MC baseline plot: {mc_plot_path}")
    print(f"Updated top variants:   {top_path}")

    return mc_plot_path, top_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="results/models_structural/rf_structural_1EY0A.pkl")
    parser.add_argument(
        "--singles_csv",
        type=str,
        default="data/processed/updated_1EY0A_struct.csv",
    )
    parser.add_argument("--mut_size", type=int, default=2, choices=[2, 3])
    parser.add_argument("--iters", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_variant_file", type=str)
    args = parser.parse_args()

    run_mc(args)


if __name__ == "__main__":
    main()