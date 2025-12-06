#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genetic Algorithm (GA) for multi-mutant design using a trained ML surrogate.

Now supports BOTH biochemical and structural features:
- Whatever numeric features the model was trained on (biochemical + structural)
  are pulled from the single-mutant table and composed for multi-mutants.

Key assumptions:
- best_model.pkl was trained on single mutants from `single_mut_clean.csv`.
- Each row has:
    * identifiers: protein_id, position, wt, mut (column names inferred)
    * numeric features: biochemical + structural (same columns used in training)
- For a multi-mutant, we compose features by SUMMING per-mutation feature vectors.
  This is appropriate if features are deltas or “effect per mutation” descriptors.

Outputs (under results/ga_mc/):
  - ga_convergence.png : best-so-far ΔΔG per generation
  - top_variants.csv   : top-N stabilizing variants discovered by GA

Run example (from repo root):
  python -m src.genetic_algorithm \
      --model_path results/models/best_model.pkl \
      --singles_csv data/processed/single_mut_clean.csv \
      --mut_size 2 \
      --pop_size 120 --ngen 60 --cxpb 0.8 --mutpb 0.2 \
      --seed 42 --top_n 50

Environment:
  - Python 3.10
  - Requires: deap, pandas, numpy, scikit-learn, matplotlib, joblib
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
from deap import base, creator, tools

# -------------------------------------------------------------------
# Helpers: paths, column inference, feature handling
# -------------------------------------------------------------------

NON_FEATURE_COLS_CANDIDATES = {
    # identifiers / metadata that should NOT be sent to the model
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
    # fallback: first non-numeric column
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            return c
    return df.columns[0]

def infer_position_col(df: pd.DataFrame) -> str:
    for c in ["position", "pos", "Position", "site"]:
        if c in df.columns:
            return c
    # fallback: first integer column
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
    Select numeric columns that serve as model inputs.
    This will naturally capture BOTH biochemical and structural features,
    as long as they are numeric and not obvious metadata.
    """
    # numeric_cols = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
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

    feat_cols = [c for c in numeric_cols if c not in NON_FEATURE_COLS_CANDIDATES]

    print(numeric_cols)

    # If everything is filtered out (edge-case), fall back to all numeric cols.
    return feat_cols or numeric_cols

def load_model(model_path: str):
    """
    Load best_model.pkl. If it is a scikit-learn Pipeline / estimator
    with feature_names_in_, we use that to enforce consistent feature order.
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
    feat_cols: list
):
    """
    Build fast lookup structures:
      - key_to_feat: mutation key -> feature vector (numpy array of len = n_features)
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
            "mut": mut
        }
        protein_to_positions[protein_id].add(pos)
        protein_pos_to_keys[(protein_id, pos)].append(key)

    protein_to_positions = {
        p: sorted(list(pos_set)) for p, pos_set in protein_to_positions.items()
    }

    return key_to_feat, protein_to_positions, protein_pos_to_keys, key_meta

def compose_features(keys, key_to_feat):
    """
    Compose single-mutation features into a multi-mutant feature vector.

    Currently: elementwise SUM across all mutations.
    This is appropriate when features are Δ-style biochemical + structural effects.

    If later you want different aggregation for some structural features (e.g., mean or max),
    you can extend this function with a per-feature aggregation scheme.
    """
    feat_list = [key_to_feat[k] for k in keys]
    return np.sum(feat_list, axis=0)


def valid_individual(ind, key_meta) -> bool:
    """
    Constraints:
      - All mutations from the SAME protein
      - Distinct positions (no double-hit at same site)
    """
    proteins = {key_meta[k]["protein_id"] for k in ind}
    if len(proteins) != 1:
        return False
    positions = [key_meta[k]["pos"] for k in ind]
    return len(set(positions)) == len(positions)

def random_individual(
    mut_size: int,
    eligible_proteins: list,
    protein_to_positions: dict,
    protein_pos_to_keys: dict,
    rng: random.Random
):
    """
    Sample a random valid genome:
      - Choose a protein with at least mut_size positions;
      - Sample mut_size distinct positions;
      - At each position, sample one mutant key.
    """
    protein_id = rng.choice(eligible_proteins)
    pos_list = protein_to_positions[protein_id]
    if len(pos_list) < mut_size:
        return None

    chosen_positions = rng.sample(pos_list, mut_size)
    keys = []
    for pos in chosen_positions:
        cand = protein_pos_to_keys[(protein_id, pos)]
        keys.append(rng.choice(cand))

    # Canonical ordering to reduce duplicates in GA search space.
    return tuple(sorted(keys))

def mutate_individual(
    ind,
    key_meta,
    protein_to_positions,
    protein_pos_to_keys,
    mutpb_site: float,
    rng: random.Random
):
    """
    Mutate an individual:
      - For each mutation "site" in the genome, with probability mutpb_site:
          * Either change the amino acid at that position, OR
          * Move the mutation to a new position in the same protein (if free pos exists).
    """
    protein_id = key_meta[ind[0]]["protein_id"]
    current = list(ind)

    for i in range(len(current)):
        if rng.random() < mutpb_site:
            # 50%: change amino acid at same position
            if rng.random() < 0.5:
                pos = key_meta[current[i]]["pos"]
                cand = protein_pos_to_keys[(protein_id, pos)]
                current[i] = rng.choice(cand)
            else:
                # 50%: move to a different position (if available)
                all_positions = protein_to_positions[protein_id]
                current_positions = {key_meta[k]["pos"] for k in current}
                free_positions = [p for p in all_positions if p not in current_positions]
                if free_positions:
                    new_pos = rng.choice(free_positions)
                    cand = protein_pos_to_keys[(protein_id, new_pos)]
                    current[i] = rng.choice(cand)
                else:
                    # No free positions: fall back to changing AA at existing position
                    pos = key_meta[current[i]]["pos"]
                    cand = protein_pos_to_keys[(protein_id, pos)]
                    current[i] = rng.choice(cand)

    new_tuple = tuple(sorted(current))
    # ✅ Return a NEW Individual, not a plain tuple
    return (creator.Individual(new_tuple),)

def repair_individual(
    ind,
    key_meta,
    protein_to_positions,
    protein_pos_to_keys,
    mut_size,
    rng
):
    """
    Enforce constraints:
      - All mutations from SAME protein
      - DISTINCT positions
      - Exactly mut_size mutations
    If violated: resample minimally until constraints satisfied.
    """

    # Extract metadata
    protein_id = key_meta[ind[0]]["protein_id"]
    current_positions = {key_meta[k]["pos"] for k in ind}

    # If we have duplicates, or not enough unique positions → fix
    if len(current_positions) != mut_size:
        all_positions = protein_to_positions[protein_id]
        
        # Pick mut_size unique positions
        if len(all_positions) < mut_size:
            raise ValueError(
                f"Protein {protein_id} has insufficient unique positions for mut_size={mut_size}."
            )
        
        chosen_positions = rng.sample(all_positions, mut_size)
        
        # Sample new keys for each chosen position
        new_keys = []
        for pos in chosen_positions:
            cand = protein_pos_to_keys[(protein_id, pos)]
            new_keys.append(rng.choice(cand))

        return creator.Individual(tuple(sorted(new_keys)))
    
    # If all constraints are OK, return a clone
    return creator.Individual(tuple(sorted(ind)))

# def mutate_individual(
#     ind,
#     key_meta,
#     protein_to_positions,
#     protein_pos_to_keys,
#     mutpb_site: float,
#     rng: random.Random
# ):
#     """
#     Mutate an individual:
#       - For each mutation "site" in the genome, with probability mutpb_site:
#           * Either change the amino acid at that position, OR
#           * Move the mutation to a new position in the same protein (if free pos exists).
#     """
#     protein_id = key_meta[ind[0]]["protein_id"]
#     current = list(ind)

#     for i in range(len(current)):
#         if rng.random() < mutpb_site:
#             # 50%: change amino acid at same position
#             if rng.random() < 0.5:
#                 pos = key_meta[current[i]]["pos"]
#                 cand = protein_pos_to_keys[(protein_id, pos)]
#                 current[i] = rng.choice(cand)
#             else:
#                 # 50%: move to a different position (if available)
#                 all_positions = protein_to_positions[protein_id]
#                 current_positions = {key_meta[k]["pos"] for k in current}
#                 free_positions = [p for p in all_positions if p not in current_positions]
#                 if free_positions:
#                     new_pos = rng.choice(free_positions)
#                     cand = protein_pos_to_keys[(protein_id, new_pos)]
#                     current[i] = rng.choice(cand)
#                 else:
#                     # No free positions: fall back to changing AA at existing position
#                     pos = key_meta[current[i]]["pos"]
#                     cand = protein_pos_to_keys[(protein_id, pos)]
#                     current[i] = rng.choice(cand)

#     new_ind = tuple(sorted(current))
#     return (new_ind,)

# -------------------------------------------------------------------
# GA core
# -------------------------------------------------------------------

import time

def run_ga(args):
    start_t = time.time_ns()
    outdir = os.path.join("results", "ga_mc")
    ensure_dir(outdir)

        # --- Load data and model ---
    singles = pd.read_csv(args.singles_csv)

    protein_col = infer_protein_col(singles)
    pos_col = infer_position_col(singles)
    wt_col, mut_col = infer_wt_mut_cols(singles)

    # Load model
    model, model_feats = load_model(args.model_path)

    if model_feats is not None:
        # Enforce exact same feature set as training
        missing = set(model_feats) - set(singles.columns)
        extra   = set(singles.columns) - set(model_feats)

        if missing:
            raise ValueError(
                "Your singles CSV is missing features used during training.\n"
                f"Missing columns: {sorted(missing)}\n"
                "Either (1) point GA to the same processed feature file used for training, "
                "or (2) update single_mut_clean.csv to include these columns."
            )

        feat_cols = list(model_feats)  # same order as training
    else:
        # Fallback: no feature_names_in_ on the model → infer numeric columns
        feat_cols = select_feature_cols(singles)
        print(feat_cols, singles)

    # At this point, len(feat_cols) should equal what StandardScaler expects
    print(f"[GA] Using {len(feat_cols)} features: {feat_cols}")

    # protein_col = infer_protein_col(singles)
    # pos_col = infer_position_col(singles)
    # wt_col, mut_col = infer_wt_mut_cols(singles)

    # feat_cols = select_feature_cols(singles)
    # model, model_feats = load_model(args.model_path)

    # print(model_feats)

    # If model remembers feature_names_in_, restrict & order cols accordingly
    if model_feats is not None:
        feat_cols = [c for c in model_feats if c in singles.columns]

    key_to_feat, protein_to_positions, protein_pos_to_keys, key_meta = make_lookup(
        singles, protein_col, pos_col, wt_col, mut_col, feat_cols
    )

    # Eligible proteins must have at least mut_size distinct positions
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

    # --- DEAP setup: minimize predicted ΔΔG (lower = better) ---
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", tuple, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    def init_individual():
        for _ in range(50):
            genome = random_individual(
                args.mut_size,
                eligible_proteins,
                protein_to_positions,
                protein_pos_to_keys,
                rng
            )
            if genome is not None and valid_individual(genome, key_meta):
                return creator.Individual(genome)
        raise RuntimeError("Failed to create a valid initial individual after 50 attempts.")

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Fitness function: use biochemical + structural features jointly
    def evaluate(ind):
        x_multi = compose_features(ind, key_to_feat).reshape(1, -1)
        pred_ddg = float(model.predict(x_multi)[0])  # scalar ΔΔG
        return (pred_ddg,)

    toolbox.register("evaluate", evaluate)

    def mate(ind1, ind2):
        """
        Simple one-point-like recombination: swap tails and repair if needed.
        """
        k = rng.randrange(1, len(ind1))
        child1 = tuple(sorted(ind1[:k] + ind2[k:]))
        child2 = tuple(sorted(ind2[:k] + ind1[k:]))

        # Light repair via heavy mutation if constraints violated
        for child in (child1, child2):
            if not valid_individual(child, key_meta):
                child, = mutate_individual(
                    child,
                    key_meta,
                    protein_to_positions,
                    protein_pos_to_keys,
                    mutpb_site=0.9,
                    rng=rng
                )
        return creator.Individual(child1), creator.Individual(child2)

    toolbox.register(
        "mutate",
        mutate_individual,
        key_meta=key_meta,
        protein_to_positions=protein_to_positions,
        protein_pos_to_keys=protein_pos_to_keys,
        mutpb_site=args.mutpb_site,
        rng=rng
    )
    toolbox.register("mate", mate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # --- Initialize population ---
    population = toolbox.population(n=args.pop_size)
    hof = tools.HallOfFame(maxsize=max(args.top_n, 10))

    # For logging
    best_curve = []

    # Initial evaluation
    start = time.perf_counter()
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    hof.update(population)
    best_curve.append(hof[0].fitness.values[0])

    # --- Evolution loop (manual variation, no tools.varAnd, no in-place slice) ---
    for gen in range(1, args.ngen + 1):
        # Selection
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < args.cxpb:
                child1, child2 = toolbox.mate(offspring[i], offspring[i + 1])
                
                child1 = repair_individual(
                    child1,
                    key_meta,
                    protein_to_positions,
                    protein_pos_to_keys,
                    args.mut_size,
                    rng
                )

                child2 = repair_individual(
                    child2,
                    key_meta,
                    protein_to_positions,
                    protein_pos_to_keys,
                    args.mut_size,
                    rng
                )

                offspring[i], offspring[i + 1] = child1, child2
                # invalidate fitness
                if hasattr(offspring[i], "fitness"):
                    del offspring[i].fitness.values
                if hasattr(offspring[i + 1], "fitness"):
                    del offspring[i + 1].fitness.values

        # Mutation
        for i, mutant in enumerate(offspring):
            if random.random() < args.mutpb:
                new_ind, = toolbox.mutate(mutant)

                repaired = repair_individual(
                    new_ind,
                    key_meta,
                    protein_to_positions,
                    protein_pos_to_keys,
                    args.mut_size,
                    rng
                )

                offspring[i] = repaired
                if hasattr(offspring[i], "fitness"):
                    del offspring[i].fitness.values

        # Evaluate invalid offspring
        invalid_indices = [i for i, ind in enumerate(offspring) if not ind.fitness.valid]
        for i in invalid_indices:
            ind = offspring[i]
            if not valid_individual(ind, key_meta):
                repaired = repair_individual(
                    ind,
                    key_meta,
                    protein_to_positions,
                    protein_pos_to_keys,
                    args.mut_size,
                    rng
                )
                offspring[i] = repaired
                ind = repaired
            ind.fitness.values = toolbox.evaluate(ind)

        population = offspring
        hof.update(population)

        best_curve.append(hof[0].fitness.values[0])

        if gen % max(1, args.ngen // 10) == 0:
            best = hof[0].fitness.values[0]
            print(f"[Gen {gen:>3}] best ΔΔG = {best:.3f} kcal/mol")

    elapsed = time.perf_counter() - start
    best_final = hof[0].fitness.values[0]
    print(f"GA finished in {elapsed:.2f}s. Best predicted ΔΔG = {best_final:.3f} kcal/mol")

    # --- Plot convergence ---
    plt.figure(figsize=(6, 4))
    plt.plot(best_curve, linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Best-so-far predicted ΔΔG (kcal/mol) ↓")
    plt.title(f"GA Convergence (mut_size = {args.mut_size})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ga_plot_path = os.path.join(outdir, "ga_convergence.png")
    plt.savefig(ga_plot_path, dpi=200)
    plt.close()

    # --- Export top variants ---
    rows = []
    for ind in hof:
        pred_ddg = ind.fitness.values[0]
        protein_id = next(iter({key_meta[k]["protein_id"] for k in ind}))
        pos_list = [key_meta[k]["pos"] for k in ind]
        muts = [f"{key_meta[k]['wt']}>{key_meta[k]['mut']}" for k in ind]
        pretty = [f"{p}{m}" for p, m in sorted(zip(pos_list, muts))]

        rows.append({
            "protein_id": protein_id,
            "mutations": ";".join(ind),               # canonical keys
            "pretty_sites": ";".join(map(str, pretty)),
            "num_mutations": len(ind),
            "predicted_ddg": pred_ddg,
            "method": "GA"
        })

    df_top = (
        pd.DataFrame(rows)
        .sort_values("predicted_ddg", ascending=True)
        .head(args.top_n)
    )
    if args.out_variant_file:
        top_path = args.out_variant_file
    else:
        top_path = os.path.join(outdir, "top_variants.csv")
    df_top.to_csv(top_path, index=False)

    print(f"Saved GA convergence plot: {ga_plot_path}")
    print(f"Saved GA top variants:     {top_path}")

    print(f"Total time for Genetic Algo: {(time.time_ns()-start_t)/10**9} seconds")

    return ga_plot_path, top_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="results/models_structural/rf_structural_1EY0A.pkl")
    parser.add_argument("--singles_csv", type=str, default="data/processed/updated_1EY0A_struct.csv")
    parser.add_argument("--mut_size", type=int, default=2, choices=[2, 3])
    parser.add_argument("--pop_size", type=int, default=120)
    parser.add_argument("--ngen", type=int, default=60)
    parser.add_argument("--cxpb", type=float, default=0.8)
    parser.add_argument("--mutpb", type=float, default=0.2)
    parser.add_argument(
        "--mutpb_site",
        type=float,
        default=0.3,
        help="Per-site mutation probability inside mutate_individual()."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--out_variant_file", type=str, default="results/ga_mc/top_variants.csv")
    args = parser.parse_args()

    run_ga(args)


if __name__ == "__main__":
    main()