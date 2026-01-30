#!/usr/bin/env python3
"""
Canonicalize antibiotic names for phenotype data.

This MUST be run before:
- filtering antibiotics
- external validation
- metric computation

Output:
data/merged/all_phenotypes_canonical.csv
"""

import pandas as pd
import re

IN_PATH = "data/merged/all_phenotypes_final.csv"
OUT_PATH = "data/merged/all_phenotypes_canonical.csv"

# -----------------------------
# Canonical mapping (expert-defined)
# -----------------------------
CANONICAL = {
    # TMP/SMX variants
    "trimethoprim/sulfamethoxazole": "trimethoprim/sulfamethoxazole",
    "trimethoprim-sulfamethoxazole": "trimethoprim/sulfamethoxazole",
    "trimethoprim/sulfobactam": "trimethoprim/sulfamethoxazole",

    # beta-lactam combos
    "piperacillin-tazobactam": "piperacillin/tazobactam",
    "piperacillin/tazobactam": "piperacillin/tazobactam",
    "amoxicillin-clavulanic acid": "amoxicillin/clavulanate",
    "amoxicillin/clavulanate": "amoxicillin/clavulanate",
    "ticarcillin-clavulanic acid": "ticarcillin/clavulanate",
    "ticarcillin/clavulanate": "ticarcillin/clavulanate",

    # spelling cleanup
    "polymyxin b": "polymyxin",
    "polymyxin": "polymyxin",

    # cephalosporin variants
    "ceftazidime-avibactam": "ceftazidime/avibactam",
    "ceftazidime/avibactam": "ceftazidime/avibactam",
}


def normalize(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def main():
    df = pd.read_csv(IN_PATH)

    df["antibiotic_raw"] = df["antibiotic"]
    df["antibiotic"] = df["antibiotic"].map(normalize)

    df["antibiotic"] = df["antibiotic"].apply(
        lambda x: CANONICAL.get(x, x)
    )

    # sanity check
    print("Antibiotic counts after canonicalization:")
    print(df["antibiotic"].value_counts().head(20))

    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
