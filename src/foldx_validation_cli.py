#!/usr/bin/env python
"""
FoldX validation for GA/MC multi-mutants using ONLY the FoldX CLI.

Pipeline:
  1. Take top variants from results/ga_mc/top_variants.csv
  2. For each protein_id:
       a. Copy its WT PDB into the work directory
       b. Run FoldX RepairPDB
       c. Build a single individual_list file with ALL mutants
       d. Run FoldX BuildModel once
       e. Parse Dif_<tag>.fxout for ΔΔG per mutant
  3. Merge FoldX ΔΔG with ML predictions and save to results/foldx/foldx_validation.csv

Assumes:
  - FoldX binary is callable as 'Foldx' or via the --foldx-binary argument
  - rotabase.txt is reachable (ideally in the same folder as Foldx)
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import List

import pandas as pd
import time


# ---------- Helpers ----------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], cwd: Path):
    print(f"[CMD] ({cwd})", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print("[STDOUT]\n", result.stdout)
        print("[STDERR]\n", result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def build_foldx_mutation_code(pretty_sites: str, chain_id: str = "A") -> str:
    """
    Convert pretty_sites like '12A>G;39V>G' into FoldX individual_list code:
        'AA12G,VA39G;'
    """
    muts = []
    for mut_str in pretty_sites.split(";"):
        mut_str = mut_str.strip()
        if not mut_str:
            continue

        # Format: '12A>G'
        pos_digits = ""
        i = 0
        while i < len(mut_str) and mut_str[i].isdigit():
            pos_digits += mut_str[i]
            i += 1

        pos = int(pos_digits)
        wt = mut_str[i]          # e.g. 'A'
        assert mut_str[i + 1] == ">"
        mut = mut_str[i + 2]     # e.g. 'G'

        code = f"{wt}{chain_id}{pos}{mut}"
        muts.append(code)

    return ",".join(muts) + ";"


def repair_pdb_with_foldx(
    foldx_bin: str,
    pdb_path: Path,
    work_dir: Path,
) -> Path:
    """
    Copy WT PDB into work_dir and run FoldX RepairPDB on it.
    Returns path to repaired PDB.
    """
    ensure_dir(work_dir)

    pdb_basename = pdb_path.name
    pdb_in_work = work_dir / pdb_basename

    if pdb_in_work.resolve() != pdb_path.resolve():
        print(f"[INFO] Copying WT PDB to work dir: {pdb_in_work}")
        pdb_in_work.write_bytes(pdb_path.read_bytes())

    cmd = [
        foldx_bin,
        "--command=RepairPDB",
        f"--pdb={pdb_basename}",
    ]
    run_cmd(cmd, cwd=work_dir)

    stem = pdb_in_work.stem
    repaired_name = f"{stem}_Repair.pdb"
    repaired_path = work_dir / repaired_name

    if not repaired_path.exists():
        raise FileNotFoundError(
            f"Expected repaired PDB '{repaired_path}' not found after RepairPDB."
        )
    print(f"[INFO] Repaired PDB: {repaired_path}")
    return repaired_path

def run_buildmodel(
    foldx_bin: str,
    repaired_pdb: Path,
    individual_list: Path,
    work_dir: Path,
    tag: str,
) -> Path:
    """
    Run FoldX BuildModel with a given repaired PDB and individual_list file.

    Different FoldX versions name the output Dif_*.fxout slightly differently,
    so instead of assuming Dif_<tag>.fxout, we glob for any Dif_*.fxout created
    in this work_dir after the run.
    """
    # Clean any old Dif_*.fxout to avoid mixing runs
    for f in work_dir.glob("Dif_*.fxout"):
        f.unlink()

    cmd = [
        foldx_bin,
        "--command=BuildModel",
        f"--pdb={repaired_pdb.name}",
        f"--mutant-file={individual_list.name}",
        "--numberOfRuns=1",
        f"--output-file={tag}",
    ]
    run_cmd(cmd, cwd=work_dir)

    # Look for new Dif_*.fxout files
    candidates = list(work_dir.glob("Dif_*.fxout"))
    if not candidates:
        raise FileNotFoundError(
            f"No Dif_*.fxout files found in {work_dir} after BuildModel. "
            "Check FoldX STDOUT/STDERR for errors."
        )

    if len(candidates) > 1:
        print("[WARN] Multiple Dif_*.fxout files found:")
        for c in candidates:
            print("   ", c.name)
        print("       Using the first one.")

    dif_file = candidates[0]
    print(f"[INFO] Using FoldX ΔΔG file: {dif_file}")
    return dif_file

# def run_buildmodel(
#     foldx_bin: str,
#     repaired_pdb: Path,
#     individual_list: Path,
#     work_dir: Path,
#     tag: str,
# ) -> Path:
#     """
#     Run FoldX BuildModel with a given repaired PDB and individual_list file.

#     Produces Dif_<tag>.fxout, which we will parse.
#     """
#     cmd = [
#         foldx_bin,
#         "--command=BuildModel",
#         f"--pdb={repaired_pdb.name}",
#         f"--mutant-file={individual_list.name}",
#         f"--numberOfRuns=1",
#         f"--output-file={tag}",
#     ]
#     run_cmd(cmd, cwd=work_dir)

#     dif_file = work_dir / f"Dif_{tag}.fxout"
#     if not dif_file.exists():
#         raise FileNotFoundError(
#             f"Expected FoldX ΔΔG output '{dif_file}' not found."
#         )
#     return dif_file

def parse_dif_fxout(dif_file: Path) -> list[float]:
    """
    Parse Dif_<tag>.fxout and extract ΔΔG for each mutant.

    FoldX Dif file columns are typically:
        Pdb    Group    Model_1    Average    SD

    We want the 'Average' column (mean ΔΔG). If headers are present,
    we detect the index by name; otherwise we assume the 4th column.
    """
    ddgs: list[float] = []
    ddg_col_idx = None

    with dif_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Header line: detect column index for 'Average'
            if line.lower().startswith("pdb") or line.startswith("Pdb"):
                parts = line.split("\t")
                # Try to find 'Average' column
                for i, col in enumerate(parts):
                    if col.strip().lower() == "average":
                        ddg_col_idx = i
                        break
                # Fallback if we didn't find it: typical index is 3
                if ddg_col_idx is None:
                    ddg_col_idx = 3
                continue

            # Comment lines
            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if ddg_col_idx is None:
                # No header encountered, assume 4th column is ΔΔG
                ddg_col_idx = 3

            try:
                ddg = float(parts[ddg_col_idx])
                ddgs.append(ddg)
            except Exception:
                continue

    if not ddgs:
        raise ValueError(f"No ΔΔG values parsed from {dif_file}")

    # Often the first line is WT (0.0); we drop it and keep the mutants
    if len(ddgs) > 1:
        return ddgs[1:]
    return ddgs

# def parse_dif_fxout(dif_file: Path) -> List[float]:
#     """
#     Parse Dif_<tag>.fxout and extract ΔΔG (last column) for each mutant.

#     The first non-header line is usually WT; we skip it, then parse one line per mutant.
#     """
#     ddgs = []
#     with dif_file.open() as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith("#") or line.lower().startswith("pdb"):
#                 continue
#             parts = line.split("\t")
#             try:
#                 ddg = float(parts[-1])
#                 ddgs.append(ddg)
#             except Exception:
#                 continue

#     if not ddgs:
#         raise ValueError(f"No ΔΔG values parsed from {dif_file}")

#     # Often first one is WT; we drop it and keep the rest
#     if len(ddgs) > 1:
#         return ddgs[1:]
#     return ddgs


# ---------- Main workflow ----------

def main():
    parser = argparse.ArgumentParser(
        description="FoldX CLI-based validation of GA/MC multi-mutants."
    )
    parser.add_argument(
        "--top-variants",
        required=True,
        help="Path to CSV with GA/MC variants (e.g., results/ga_mc/top_variants.csv)",
    )
    parser.add_argument(
        "--pdb-dir",
        required=True,
        help="Directory containing WT PDB files (e.g., data/structures)",
    )
    parser.add_argument(
        "--out-dir",
        default="results/foldx",
        help="Output directory for FoldX runs and validation CSV",
    )
    parser.add_argument(
        "--foldx-bin",
        default="Foldx",
        help="FoldX executable name or path (default: 'Foldx')",
    )
    parser.add_argument(
        "--chain-id",
        default="A",
        help="Chain ID to use in FoldX mutation codes (default: A)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of most-stabilizing variants to validate (by predicted_ddg)",
    )

    args = parser.parse_args()

    foldx_bin = args.foldx_bin
    pdb_dir = Path(args.pdb_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # 1) Load variants
    df = pd.read_csv(args.top_variants)

    required_cols = {
        "protein_id",
        "mutations",
        "pretty_sites",
        "num_mutations",
        "predicted_ddg",
        "method",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Keep top-K most stabilizing (most negative ΔΔG)
    df_sorted = df.sort_values("predicted_ddg").head(args.top_k).reset_index(drop=True)

    all_results = []

    # 2) Group by protein_id (you currently have only 1EY0A, but this is future-proof)
    for protein_id, group in df_sorted.groupby("protein_id"):
        start_t = time.time_ns()
        print(f"\n[Protein] {protein_id}")

        protein_id = protein_id[:len(protein_id)-1]

        pdb_path = pdb_dir / f"{protein_id}.pdb"
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB not found for {protein_id}: {pdb_path}")

        # Work directory per protein
        work_dir = out_dir / f"{protein_id}_foldx"
        ensure_dir(work_dir)

        # 2a) Repair PDB
        repaired_pdb = repair_pdb_with_foldx(
            foldx_bin=foldx_bin,
            pdb_path=pdb_path,
            work_dir=work_dir,
        )

        # 2b) Write individual_list file with ALL mutants for this protein
        ind_file = work_dir / "individual_list.txt"
        lines = []
        for _, row in group.iterrows():
            pretty_sites = str(row["pretty_sites"])
            line = build_foldx_mutation_code(pretty_sites, chain_id=args.chain_id)
            lines.append(line)

        with ind_file.open("w") as f:
            for line in lines:
                f.write(line + "\n")

        # 2c) Run BuildModel once
        tag = f"{protein_id}_GA_MC"
        dif_file = run_buildmodel(
            foldx_bin=foldx_bin,
            repaired_pdb=repaired_pdb,
            individual_list=ind_file,
            work_dir=work_dir,
            tag=tag,
        )

        # 2d) Parse ΔΔG
        ddgs = parse_dif_fxout(dif_file)
        if len(ddgs) != len(group):
            print(
                f"[WARN] #ΔΔG ({len(ddgs)}) != #variants ({len(group)}) for {protein_id}. "
                "Truncating to min length."
            )
        n = min(len(ddgs), len(group))
        sub = group.iloc[:n].copy()
        sub["ddg_foldx"] = ddgs[:n]
        sub["ddg_diff_model_minus_foldx"] = sub["predicted_ddg"] - sub["ddg_foldx"]
        sub["time_taken"] = time.time_ns() - start_t
        all_results.append(sub)

    if not all_results:
        print("[INFO] No results produced.")
        return

    df_out = pd.concat(all_results, ignore_index=True)
    out_csv = out_dir / "foldx_validation.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved FoldX validation results to {out_csv}")


if __name__ == "__main__":
    main()