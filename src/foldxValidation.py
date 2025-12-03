# import pyfoldx

# def getFoldxDDG(protein_id: str, mutations: list):
#     protein_structure = pyfoldx.structure.Structure("data/1EY0.pdb")
#     for mutation in mutations:




# # Load a protein structure
# protein_structure = pyfoldx.structure.Structure("my_protein.pdb")

# # Define a mutation (e.g., Alanine at position 10 in chain A to Glycine)
# mutation = "A10A>G"

# # Apply the mutation and calculate stability change
# mutated_structure, ddg_value = protein_structure.mutate(mutation)

# # Print the calculated ddG value
# print(f"ddG for mutation {mutation}: {ddg_value}")


#!/usr/bin/env python
"""
FoldX validation of GA-selected multi-mutants using pyFoldX.

Usage (from repo root):

    (proteinML) python -m src.validation_foldx \
        --top-variants results/ga_mc/top_variants.csv \
        --pdb-id 1EY0A \
        --pdb-path data/structures/1EY0A.pdb \
        --chain-id A \
        --out-dir results/foldx \
        --n-runs 3
"""

import argparse
import csv
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

# ---- pyFoldX imports ----
try:
    from pyfoldx.structure import Structure
except ImportError as e:
    raise ImportError(
        "pyfoldx is not installed or not importable. "
        "Install it and make sure FOLDX_LOCATION is set."
    ) from e


# -------------------------
# Data model
# -------------------------

@dataclass
class Variant:
    protein_id: str
    pretty_sites: str      # e.g. "39V>G;94A>G;128S>G"
    num_mutations: int
    predicted_ddg_ml: float
    method: str            # "GA" or "MC" etc.


# -------------------------
# Parsing utilities
# -------------------------

def load_top_variants(csv_path: str) -> List[Variant]:
    """
    Load GA/MC top variants from CSV.

    Expected columns:
      - protein_id
      - pretty_sites
      - num_mutations
      - predicted_ddg
      - method
    """
    df = pd.read_csv(csv_path)
    variants: List[Variant] = []

    for _, row in df.iterrows():
        variants.append(
            Variant(
                protein_id=str(row["protein_id"]),
                pretty_sites=str(row["pretty_sites"]),
                num_mutations=int(row["num_mutations"]),
                predicted_ddg_ml=float(row["predicted_ddg"]),
                method=str(row.get("method", "GA")),
            )
        )

    return variants


def parse_pretty_sites_to_foldx_codes(pretty_sites: str, chain_id: str) -> list[str]:
    """
    Convert pretty_sites like:
        '12A>G;39V>G'
    into FoldX codes like:
        ['AA12G', 'VA39G']  (for chain_id='A')
    """
    codes = []
    for mut_str in pretty_sites.split(";"):
        mut_str = mut_str.strip()
        if not mut_str:
            continue

        # Our format is exactly '<pos><WTaa>><Mutaa>', e.g. '12A>G'
        pos_digits = ""
        i = 0
        while i < len(mut_str) and mut_str[i].isdigit():
            pos_digits += mut_str[i]
            i += 1

        pos = int(pos_digits)
        wt = mut_str[i]          # 'A' or 'V'
        assert mut_str[i + 1] == ">"
        mut = mut_str[i + 2]     # 'G', etc.

        code = f"{wt}{chain_id}{pos}{mut}"
        codes.append(code)

    return codes

# def parse_pretty_sites_to_foldx_codes(
#     pretty_sites: str,
#     chain_id: str
# ) -> List[str]:
#     """
#     Convert a pretty_sites string like:
#         "39V>G;94A>G;128S>G"
#     into FoldX "individual_list.txt" mutation codes:
#         ["VA39G", "AA94G", "SA128G"]

#     Assumes pattern "<pos><WTaa>><Mutaa>" (no chain in the pretty_sites),
#     or "1EY0A:<pos><WTaa>><Mutaa>" if prefixed with protein id.
#     """
#     codes: List[str] = []

#     for mut_str in pretty_sites.split(";"):
#         mut_str = mut_str.strip()
#         if not mut_str:
#             continue

#         # Strip protein prefix if present: "1EY0A:39V>G" -> "39V>G"
#         if ":" in mut_str:
#             _, mut_core = mut_str.split(":", 1)
#         else:
#             mut_core = mut_str

#         # mut_core is like "39V>G"
#         try:
#             # position = all leading digits
#             pos_digits = ""
#             i = 0
#             while i < len(mut_core) and mut_core[i].isdigit():
#                 pos_digits += mut_core[i]
#                 i += 1

#             pos = int(pos_digits)
#             wt = mut_core[i]      # e.g. 'V'
#             assert mut_core[i + 1] == ">"
#             mut = mut_core[i + 2]  # e.g. 'G'

#         except Exception as e:
#             raise ValueError(
#                 f"Could not parse mutation '{mut_str}' from pretty_sites '{pretty_sites}'"
#             ) from e

#         # FoldX code: WT, chain, pos, mutant; no spaces.
#         code = f"{wt}{chain_id}{pos}{mut}"
#         codes.append(code)

#     return codes


# -------------------------
# FoldX helpers
# -------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_foldx(
    command: str,
    pdb_name: str,
    work_dir: str,
    mutant_file: str | None = None,
    n_runs: int = 1,
) -> None:
    """
    Run a FoldX command via subprocess.
    Requires FoldX to be in PATH or FOLDX_LOCATION to be set appropriately.
    """
    cfg_parts = [
        f"--command={command}",
        f"--pdb={pdb_name}",
        f"--numberOfRuns={n_runs}",
    ]
    if mutant_file is not None:
        cfg_parts.append(f"--mutant-file={mutant_file}")

    cmd = ["FoldX"] + cfg_parts

    print(f"[FoldX] Running: {' '.join(cmd)} in {work_dir}")

    result = subprocess.run(
        cmd,
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print("[FoldX] STDOUT:\n", result.stdout)
        print("[FoldX] STDERR:\n", result.stderr)
        raise RuntimeError(f"FoldX command failed: {' '.join(cmd)}")


def prepare_repaired_pdb(
    pdb_path: str,
    work_dir: str,
) -> str:
    """
    Copy original PDB into work_dir and run RepairPDB.
    Returns path to repaired PDB.
    """
    ensure_dir(work_dir)

    pdb_basename = os.path.basename(pdb_path)
    target_pdb = os.path.join(work_dir, pdb_basename)

    # Copy PDB
    if os.path.abspath(pdb_path) != os.path.abspath(target_pdb):
        shutil.copy2(pdb_path, target_pdb)

    # Run RepairPDB
    run_foldx(
        command="RepairPDB",
        pdb_name=pdb_basename,
        work_dir=work_dir,
        mutant_file=None,
        n_runs=1,
    )

    # By FoldX convention, repaired file is <stem>_Repair.pdb
    stem, ext = os.path.splitext(pdb_basename)
    repaired_name = f"{stem}_Repair.pdb"
    repaired_path = os.path.join(work_dir, repaired_name)

    if not os.path.exists(repaired_path):
        raise FileNotFoundError(
            f"Expected repaired PDB '{repaired_path}' not found."
        )

    return repaired_path


def write_individual_list(
    mutations_codes: List[str],
    path: str,
) -> None:
    """
    Write a single-line individual_list.txt file:
        "VA39G,AA94G,SA128G;"
    """
    line = ",".join(mutations_codes) + ";"
    with open(path, "w") as f:
        f.write(line + "\n")


def build_mutant_with_foldx(
    repaired_pdb_path: str,
    mutations_codes: List[str],
    work_dir: str,
    tag: str,
    n_runs: int = 1,
) -> str:
    """
    Run FoldX BuildModel on a repaired PDB for a given multi-mutant.
    Returns path to the mutant PDB (first run).
    """
    pdb_basename = os.path.basename(repaired_pdb_path)

    # Create a per-variant individual list file
    individual_list_name = f"individual_list_{tag}.txt"
    individual_list_path = os.path.join(work_dir, individual_list_name)
    write_individual_list(mutations_codes, individual_list_path)

    # Clean up previous model files for this PDB/tag to avoid confusion
    for fname in os.listdir(work_dir):
        if fname.endswith("_Repair_1.pdb") or fname.startswith("WT_"):
            os.remove(os.path.join(work_dir, fname))

    # Run BuildModel
    run_foldx(
        command="BuildModel",
        pdb_name=pdb_basename,
        work_dir=work_dir,
        mutant_file=individual_list_name,
        n_runs=n_runs,
    )

    # After BuildModel, we expect something like:
    #   WT_<stem>_Repair_1.pdb
    #   <stem>_Repair_1.pdb
    stem, ext = os.path.splitext(pdb_basename)
    mutant_candidate = os.path.join(work_dir, f"{stem}_Repair_1.pdb")

    if not os.path.exists(mutant_candidate):
        # Fallback: look for any *_Repair_1.pdb that isn't WT_
        candidates = [
            os.path.join(work_dir, f)
            for f in os.listdir(work_dir)
            if f.endswith("_Repair_1.pdb") and not f.startswith("WT_")
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No mutant PDB found after BuildModel for tag '{tag}'."
            )
        mutant_candidate = candidates[0]

    return mutant_candidate


# -------------------------
# pyFoldX energy evaluation
# -------------------------

def compute_total_energy(pdb_path: str, pdb_id: str) -> float:
    """
    Compute FoldX total energy using pyFoldX's Structure wrapper.
    Assumes FOLDX_LOCATION and rotabase.txt are configured.
    """
    pdb_dir = os.path.dirname(os.path.abspath(pdb_path))
    # In pyFoldX, Structure(name, path=dir) expects 'name.pdb' in that dir.
    # Here we symlink/copy to <pdb_id>.pdb if needed.
    target_name = f"{pdb_id}.pdb"
    target_path = os.path.join(pdb_dir, target_name)

    if os.path.abspath(pdb_path) != os.path.abspath(target_path):
        # Ensure the expected filename exists
        if os.path.exists(target_path):
            os.remove(target_path)
        shutil.copy2(pdb_path, target_path)

    # Instantiate Structure and compute energy
    s = Structure(pdb_id, path=pdb_dir)
    # If you want to force a re-repair here, you could call:
    # s = s.repair(verbose=False)
    total_energy = s.getTotalEnergy()
    return float(total_energy)


def compute_ddg_foldx(
    wt_pdb_path: str,
    mutant_pdb_path: str,
    pdb_id: str,
) -> float:
    """
    Compute ΔΔG = G_mutant - G_wt using pyFoldX energies.
    Negative ΔΔG => stabilizing (more negative energy).
    """
    g_wt = compute_total_energy(wt_pdb_path, pdb_id=f"{pdb_id}_WT")
    g_mut = compute_total_energy(mutant_pdb_path, pdb_id=f"{pdb_id}_MUT")
    return g_mut - g_wt


# -------------------------
# Main driver
# -------------------------

def validate_with_foldx(
    top_variants_path: str,
    pdb_id: str,
    pdb_path: str,
    chain_id: str,
    out_dir: str,
    n_runs: int = 1,
) -> None:
    """
    Main pipeline:
      1. Repair WT PDB with FoldX.
      2. For each GA/MC variant:
         - Build mutant with BuildModel.
         - Compute ΔΔG via pyFoldX.
      3. Save combined table as CSV.
    """
    ensure_dir(out_dir)
    work_dir = os.path.join(out_dir, "foldx_tmp")
    ensure_dir(work_dir)

    print(f"[INFO] Loading top variants from {top_variants_path}")
    variants = load_top_variants(top_variants_path)
    print(f"[INFO] Loaded {len(variants)} variants")

    print(f"[INFO] Preparing repaired PDB from {pdb_path}")
    repaired_pdb_path = prepare_repaired_pdb(pdb_path, work_dir)
    print(f"[INFO] Repaired PDB: {repaired_pdb_path}")

    # Precompute WT energy once
    print("[INFO] Computing WT total energy with pyFoldX")
    g_wt = compute_total_energy(repaired_pdb_path, pdb_id=f"{pdb_id}_WT_BASE")
    print(f"[INFO] WT total energy: {g_wt:.3f} kcal/mol")

    records: List[dict] = []

    for idx, var in enumerate(variants, start=1):
        print(f"\n[VARIANT {idx}/{len(variants)}] {var.pretty_sites}")

        # 1) Convert pretty_sites to FoldX codes
        codes = parse_pretty_sites_to_foldx_codes(var.pretty_sites, chain_id=chain_id)
        print(f"    FoldX mutation codes: {codes}")

        # 2) Build mutant structure with FoldX
        tag = f"var{idx}"
        mutant_pdb_path = build_mutant_with_foldx(
            repaired_pdb_path=repaired_pdb_path,
            mutations_codes=codes,
            work_dir=work_dir,
            tag=tag,
            n_runs=n_runs,
        )
        print(f"    Mutant PDB: {mutant_pdb_path}")

        # 3) Compute ΔΔG via pyFoldX (mutant - WT)
        ddg = compute_ddg_foldx(
            wt_pdb_path=repaired_pdb_path,
            mutant_pdb_path=mutant_pdb_path,
            pdb_id=f"{pdb_id}_{tag}",
        )
        print(f"    FoldX ΔΔG: {ddg:.3f} kcal/mol")

        records.append(
            {
                "protein_id": var.protein_id,
                "pretty_sites": var.pretty_sites,
                "num_mutations": var.num_mutations,
                "predicted_ddg_ml": var.predicted_ddg_ml,
                "ddg_foldx": ddg,
                "method": var.method,
            }
        )

    # Save combined table
    out_csv = os.path.join(out_dir, "foldx_validation.csv")
    print(f"\n[INFO] Saving FoldX validation results to {out_csv}")
    pd.DataFrame.from_records(records).to_csv(out_csv, index=False)
    print("[INFO] Done.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate GA-selected variants with FoldX via pyFoldX."
    )
    p.add_argument(
        "--top-variants",
        required=True,
        help="Path to CSV of top variants (GA/MC).",
    )
    p.add_argument(
        "--pdb-id",
        required=True,
        help="PDB identifier (e.g., 1EY0A).",
    )
    p.add_argument(
        "--pdb-path",
        required=True,
        help="Path to wild-type PDB file.",
    )
    p.add_argument(
        "--chain-id",
        default="A",
        help="Chain ID used in FoldX mutation codes (default: A).",
    )
    p.add_argument(
        "--out-dir",
        default="results/foldx",
        help="Output directory for FoldX validation results.",
    )
    p.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="FoldX numberOfRuns for BuildModel (default: 1).",
    )
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    validate_with_foldx(
        top_variants_path=args.top_variants,
        pdb_id=args.pdb_id,
        pdb_path=args.pdb_path,
        chain_id=args.chain_id,
        out_dir=args.out_dir,
        n_runs=args.n_runs,
    )


if __name__ == "__main__":
    main()