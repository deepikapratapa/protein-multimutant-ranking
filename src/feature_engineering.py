#!/usr/bin/env python3
"""
Feature engineering for protein single-mutant ΔΔG datasets.

Inputs
------
CSV with (at minimum) the following columns (as produced by 01_EDA.ipynb):
    - clid (optional)
    - pdb_id (string)
    - chain (string)
    - wt (single-letter amino acid)
    - res_index (int)
    - mut (single-letter amino acid)   # (this is the target AA)
    - ddg_exp_kcal (float)
    - pH (optional float)
    - temp_C (optional float)

Outputs
-------
CSV with engineered features saved to:
    data/processed/features_basic.csv   (default)  OR --out path

Features
--------
Sequence & simple biophysics:
    - pos (res_index)
    - delta_hydropathy (Kyte-Doolittle)
    - delta_charge (formal charge @ ~pH 7)
    - delta_volume (scaled Van der Waals volume)
    - delta_polarity (Grantham polarity scale)
    - blosum62 (substitution score)
    - is_gly, is_pro, is_to_gly, is_to_pro (binary flags)
    - env: pH, temp_C (if present)

Structure (if DSSP / NACCESS are available):
    - ss_coarse (E/H/C)
    - ss_idx (0 = strand, 1 = helix, 2 = coil)
    - ss_H, ss_E, ss_C (one-hot)
    - asa (absolute solvent accessible area)
    - asa_norm (per-chain normalized ASA)
    - is_buried (asa < 30 Å²)
    - is_exposed (asa > 80 Å²)
    - asa_rel, asa_abs (from NACCESS, if present)
    - ddg_foldx (from FoldX, if provided)

Label:
    - ddg_exp_kcal
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# -------------------------------
# Canonical constants / lookups
# -------------------------------

AA20 = list("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydropathy (higher = more hydrophobic)
HYDRO = {
    'A': 1.8,  'C': 2.5,  'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5,  'K': -3.9, 'L': 3.8,
    'M': 1.9,  'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2,  'W': -0.9, 'Y': -1.3
}

# Formal sidechain charge at ~pH 7 (very coarse)
CHARGE7 = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': +0.1, 'I': 0, 'K': +1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': +1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
}

# Approx. side-chain volumes (Å^3) (Zamyatnin / Chothia-style)
VOL = {
    'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
    'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
    'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
    'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6
}

# Grantham polarity (lower = nonpolar, higher = polar)
POLARITY = {
    'A': 8.1,  'C': 5.5,  'D': 13.0, 'E': 12.3, 'F': 5.2,
    'G': 9.0,  'H': 10.4, 'I': 5.2,  'K': 11.3, 'L': 4.9,
    'M': 5.7,  'N': 11.6, 'P': 8.0,  'Q': 10.5, 'R': 10.5,
    'S': 9.2,  'T': 8.6,  'V': 5.9,  'W': 5.4,  'Y': 6.2
}

# BLOSUM62 matrix (subset; full 20x20)
# Stored as nested dict for clarity; values from standard BLOSUM62.
_BLOSUM62_ROWS = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62 = [
#    A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0], # A
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3], # R
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3], # N
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3], # D
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1], # C
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2], # Q
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2], # E
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3], # G
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3], # H
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3], # I
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1], # L
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2], # K
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1], # M
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1], # F
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2], # P
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2], # S
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0], # T
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3], # W
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1], # Y
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4], # V
]
B62 = {r: {c: v for c, v in zip(_BLOSUM62_ROWS, row)}
       for r, row in zip(_BLOSUM62_ROWS, _BLOSUM62)}

# -------------------------------
# Core feature functions
# -------------------------------

def _safe_map(letter: str, table: dict[str, float]) -> float:
    if pd.isna(letter):
        return np.nan
    letter = str(letter).upper()
    return table.get(letter, np.nan)


def compute_basic_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sequence/biochem features; expects wt, mut, res_index present."""
    if not {"wt", "mut", "res_index"}.issubset(df.columns):
        raise ValueError("DataFrame must contain columns: wt, mut, res_index")

    out = df.copy()
    out["pos"] = out["res_index"].astype("Int64")

    # lookups for wt/mut
    out["wt_hydro"] = out["wt"].map(lambda a: _safe_map(a, HYDRO))
    out["mt_hydro"] = out["mut"].map(lambda a: _safe_map(a, HYDRO))
    out["delta_hydropathy"] = out["mt_hydro"] - out["wt_hydro"]

    out["wt_charge"] = out["wt"].map(lambda a: _safe_map(a, CHARGE7))
    out["mt_charge"] = out["mut"].map(lambda a: _safe_map(a, CHARGE7))
    out["delta_charge"] = out["mt_charge"] - out["wt_charge"]

    out["wt_vol"] = out["wt"].map(lambda a: _safe_map(a, VOL))
    out["mt_vol"] = out["mut"].map(lambda a: _safe_map(a, VOL))
    out["delta_volume"] = out["mt_vol"] - out["wt_vol"]

    out["wt_polar"] = out["wt"].map(lambda a: _safe_map(a, POLARITY))
    out["mt_polar"] = out["mut"].map(lambda a: _safe_map(a, POLARITY))
    out["delta_polarity"] = out["mt_polar"] - out["wt_polar"]

    # BLOSUM62 substitution score
    def blosum_score(w, m):
        if pd.isna(w) or pd.isna(m):
            return np.nan
        w, m = str(w).upper(), str(m).upper()
        return B62.get(w, {}).get(m, np.nan)

    out["blosum62"] = [blosum_score(w, m) for w, m in zip(out["wt"], out["mut"])]

    # simple flags
    out["is_gly"] = (out["wt"] == "G").astype(int)
    out["is_pro"] = (out["wt"] == "P").astype(int)
    out["is_to_gly"] = (out["mut"] == "G").astype(int)
    out["is_to_pro"] = (out["mut"] == "P").astype(int)

    # keep env if present
    for col in ["pH", "temp_C"]:
        if col in out.columns:
            out[col] = out[col]

    return out

# -------------------------------
# Optional stubs + structural features
# -------------------------------

def attach_dssp_features(df: pd.DataFrame, dssp_dir: Path | None) -> pd.DataFrame:
    """
    Merge DSSP per-residue features (e.g., SS, ASA) if CSVs exist.

    Expected per-protein file pattern (customize as needed):
        {dssp_dir}/{pdb_id}_{chain}_dssp.csv with columns:
            res_index, ss (DSSP code), asa

    This function performs a left-join on (pdb_id, chain, res_index).
    """
    if dssp_dir is None:
        return df
    dssp_dir = Path(dssp_dir)
    if not dssp_dir.exists():
        print(f"[WARN] DSSP dir not found: {dssp_dir}")
        return df

    rows = []
    for f in dssp_dir.glob("*.csv"):
        try:
            tmp = pd.read_csv(f)
            # heuristics to parse pdb/chain from filename like 1EY0_A_dssp.csv
            stem = f.stem
            parts = stem.replace("__", "_").split("_")
            pdb, chain = (parts[0], parts[1]) if len(parts) >= 2 else (None, None)
            tmp["pdb_id"] = pdb
            tmp["chain"] = chain
            rows.append(tmp)
        except Exception as e:
            print(f"[WARN] failed to read DSSP file {f}: {e}")
    if not rows:
        print("[INFO] No DSSP CSVs merged.")
        return df

    dssp_all = pd.concat(rows, ignore_index=True)
    # normalize column names
    dssp_all.columns = [c.strip().lower() for c in dssp_all.columns]
    if "res_index" not in dssp_all.columns and "resid" in dssp_all.columns:
        dssp_all = dssp_all.rename(columns={"resid": "res_index"})
    keep = [c for c in dssp_all.columns
            if c in {"pdb_id", "chain", "res_index", "ss", "asa"}]
    dssp_all = dssp_all[keep].copy()

    merged = df.merge(dssp_all, on=["pdb_id", "chain", "res_index"], how="left")
    return merged


def attach_naccess_features(df: pd.DataFrame, naccess_dir: Path | None) -> pd.DataFrame:
    """
    Merge NACCESS (ASA) per-residue features if CSVs exist.

    Expected per-protein file pattern (customize as needed):
        {naccess_dir}/{pdb_id}_{chain}_naccess.csv with:
            res_index, asa_rel, asa_abs
    """
    if naccess_dir is None:
        return df
    naccess_dir = Path(naccess_dir)
    if not naccess_dir.exists():
        print(f"[WARN] NACCESS dir not found: {naccess_dir}")
        return df

    rows = []
    for f in naccess_dir.glob("*.csv"):
        try:
            tmp = pd.read_csv(f)
            stem = f.stem
            parts = stem.replace("__", "_").split("_")
            pdb, chain = (parts[0], parts[1]) if len(parts) >= 2 else (None, None)
            tmp["pdb_id"] = pdb
            tmp["chain"] = chain
            rows.append(tmp)
        except Exception as e:
            print(f"[WARN] failed to read NACCESS file {f}: {e}")
    if not rows:
        print("[INFO] No NACCESS CSVs merged.")
        return df

    acc_all = pd.concat(rows, ignore_index=True)
    acc_all.columns = [c.strip().lower() for c in acc_all.columns]
    if "res_index" not in acc_all.columns and "resid" in acc_all.columns:
        acc_all = acc_all.rename(columns={"resid": "res_index"})
    keep = [c for c in acc_all.columns
            if c in {"pdb_id", "chain", "res_index", "asa_rel", "asa_abs"}]
    acc_all = acc_all[keep].copy()

    merged = df.merge(acc_all, on=["pdb_id", "chain", "res_index"], how="left")
    return merged


def attach_foldx_ddg(df: pd.DataFrame, foldx_csv: Path | None) -> pd.DataFrame:
    """
    Merge precomputed FoldX single-mutant ΔΔG if available.

    Expected CSV columns in foldx_csv:
        pdb_id, chain, res_index, wt, mut, ddg_foldx

    Left-joins on (pdb_id, chain, res_index, wt, mut).
    """
    if foldx_csv is None:
        return df
    foldx_csv = Path(foldx_csv)
    if not foldx_csv.exists():
        print(f"[WARN] FoldX CSV not found: {foldx_csv}")
        return df

    fdx = pd.read_csv(foldx_csv)
    fdx.columns = [c.strip().lower() for c in fdx.columns]
    # normalize expected columns
    ren = {"delta_delta_g": "ddg_foldx", "ddg": "ddg_foldx", "mut_aa": "mut"}
    fdx = fdx.rename(columns=ren)
    needed = {"pdb_id", "chain", "res_index", "wt", "mut", "ddg_foldx"}
    missing = needed - set(fdx.columns)
    if missing:
        print(f"[WARN] FoldX CSV missing columns: {missing}")
        return df

    merged = df.merge(fdx[list(needed)],
                      on=["pdb_id", "chain", "res_index", "wt", "mut"],
                      how="left")
    return merged


def compute_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive engineered structural features from DSSP / NACCESS columns.

    Expects columns (if available):
        - ss  (DSSP code)
        - asa (absolute ASA)
        - asa_rel, asa_abs (from NACCESS)
    """
    out = df.copy()

    # --- Secondary structure: map DSSP codes to coarse classes + one-hot ---
    if "ss" in out.columns:
        # ensure string and strip spaces
        out["ss"] = out["ss"].astype(str).str.strip()

        # Map full DSSP codes to coarse E/H/C
        dssp_to_coarse = {
            "H": "H", "G": "H", "I": "H",   # helices
            "E": "E", "B": "E",             # strands
            "T": "C", "S": "C", " ": "C",   # coil / turn / bend / blank
        }
        out["ss_coarse"] = out["ss"].map(lambda s: dssp_to_coarse.get(s, "C"))

        ss_idx_map = {"E": 0, "H": 1, "C": 2}
        out["ss_idx"] = out["ss_coarse"].map(ss_idx_map).astype("Int64")

        # One-hot
        out["ss_H"] = (out["ss_coarse"] == "H").astype(int)
        out["ss_E"] = (out["ss_coarse"] == "E").astype(int)
        out["ss_C"] = (out["ss_coarse"] == "C").astype(int)

    # --- ASA based features ---
    if "asa" in out.columns:
        out["asa"] = pd.to_numeric(out["asa"], errors="coerce")

        # Normalize ASA per (pdb_id, chain) if available
        if {"pdb_id", "chain"}.issubset(out.columns):
            out["asa_norm"] = out.groupby(["pdb_id", "chain"])["asa"].transform(
                lambda x: x / x.max() if x.max() > 0 else x
            )
        else:
            max_asa = out["asa"].max()
            out["asa_norm"] = out["asa"] / max_asa if max_asa and max_asa > 0 else out["asa"]

        # Simple buried / exposed flags (cutoffs can be tuned)
        out["is_buried"] = (out["asa"] < 30).astype(int)
        out["is_exposed"] = (out["asa"] > 80).astype(int)

    return out

# -------------------------------
# Orchestrator
# -------------------------------

def build_features(
    inp: Path,
    out: Path,
    dssp_dir: Path | None = None,
    naccess_dir: Path | None = None,
    foldx_csv: Path | None = None,
) -> pd.DataFrame:
    """Load clean CSV, compute basic + structural features, save to out."""
    df = pd.read_csv(inp)
    # tolerate both schemas: if original 'mut_aa' exists, rename to 'mut'
    if "mut" not in df.columns and "mut_aa" in df.columns:
        df = df.rename(columns={"mut_aa": "mut"})

    basic = compute_basic_sequence_features(df)
    basic = attach_dssp_features(basic, dssp_dir)
    basic = attach_naccess_features(basic, naccess_dir)
    basic = attach_foldx_ddg(basic, foldx_csv)
    basic = compute_structural_features(basic)

    # choose final columns
    base_cols = [
        c
        for c in [
            "clid",
            "pdb_id",
            "chain",
            "wt",
            "res_index",
            "mut",
            "ddg_exp_kcal",
            "pH",
            "temp_C",
        ]
        if c in basic.columns
    ]

    feat_cols = [
        "pos",
        "delta_hydropathy",
        "delta_charge",
        "delta_volume",
        "delta_polarity",
        "blosum62",
        "is_gly",
        "is_pro",
        "is_to_gly",
        "is_to_pro",
        # structural / physics features (only kept if present)
        "ss_idx",
        "ss_H",
        "ss_E",
        "ss_C",
        "asa",
        "asa_norm",
        "is_buried",
        "is_exposed",
        "asa_rel",
        "asa_abs",
        "ddg_foldx",
    ]

    feat_cols = [c for c in feat_cols if c in basic.columns]

    final = basic[base_cols + feat_cols].copy()
    out.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out, index=False)
    print(f"[OK] Features saved → {out}  (n={len(final)}, d={final.shape[1]})")
    return final

# -------------------------------
# CLI
# -------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Protein ΔΔG feature engineering")
    p.add_argument(
        "--in",
        dest="inp",
        required=True,
        type=Path,
        help="Input clean CSV (e.g., data/processed/single_mut_clean.csv)",
    )
    p.add_argument(
        "--out",
        dest="out",
        required=True,
        type=Path,
        help="Output CSV (e.g., data/processed/features_basic.csv)",
    )
    p.add_argument(
        "--dssp_dir",
        type=Path,
        default=None,
        help="Optional: directory of DSSP per-chain CSVs",
    )
    p.add_argument(
        "--naccess_dir",
        type=Path,
        default=None,
        help="Optional: directory of NACCESS per-chain CSVs",
    )
    p.add_argument(
        "--foldx",
        dest="foldx_csv",
        type=Path,
        default=None,
        help="Optional: FoldX single-mutant CSV",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    build_features(args.inp, args.out, args.dssp_dir, args.naccess_dir, args.foldx_csv)


if __name__ == "__main__":
    sys.exit(main())