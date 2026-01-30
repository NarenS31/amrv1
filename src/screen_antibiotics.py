#!/usr/bin/env python3
"""
Screen antibiotics for external validation suitability.

Outputs:
- external_validation_antibiotic_screen.csv
- Prints KEEP / CONDITIONAL / DROP groups with reasons

This is a HARD FILTER for serious external validation.
"""

import pandas as pd
import re

# ============================
# CONFIG (hard gates)
# ============================
MIN_R = 50
MIN_TOTAL = 200
MAX_I_FRAC = 0.15

# ============================
# Expert priors (genotype→phenotype feasibility)
# ============================

KEEP_KEYWORDS = [
    # Fluoroquinolones
    "ciprofloxacin", "levofloxacin",
    # Cephalosporins / monobactam
    "cefepime", "ceftriaxone", "ceftazidime", "aztreonam",
    # Carbapenems
    "meropenem", "imipenem", "ertapenem", "doripenem",
    # Aminoglycosides
    "gentamicin", "amikacin", "tobramycin",
    # Folate pathway
    "trimethoprim/sulfamethoxazole", "trimethoprim", "sulfamethoxazole",
]

CONDITIONAL_KEYWORDS = [
    "tetracycline", "doxycycline", "minocycline",
    "chloramphenicol",
    "cefoxitin",  # AmpC/porin mess; usable but ugly
]

DROP_KEYWORDS = [
    # polymyxins
    "polymyxin", "colistin",
    # loss-of-function / regulation heavy
    "nitrofurantoin", "fosfomycin", "tigecycline",
    # combo drugs (unfair baseline comparison)
    "avibactam", "clavulanic", "tazobactam", "sulbactam",
    # nonstandard / veterinary / garbage
    "ceftiofur", "ticarcillin", "norfloxacin",
]

# ============================
# Helpers
# ============================


def normalize_name(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def keyword_match(name: str, keywords) -> bool:
    return any(k in name for k in keywords)


def classify_antibiotic(name: str, row: pd.Series):
    I = int(row.get("I", 0))
    R = int(row.get("R", 0))
    S = int(row.get("S", 0))
    total = I + R + S

    # --- hard statistical gates ---
    if R < MIN_R:
        return "DROP", f"R<{MIN_R} (too few resistant samples)"

    if total < MIN_TOTAL:
        return "DROP", f"Total<{MIN_TOTAL} (too small / noisy)"

    if total > 0 and (I / total) > MAX_I_FRAC:
        return "CONDITIONAL", f"High I fraction ({I}/{total}={I/total:.1%}); label instability"

    # --- biology / baseline feasibility ---
    if keyword_match(name, DROP_KEYWORDS):
        return "DROP", "Unreliable genotype→phenotype mapping or unfair baseline comparison"

    if keyword_match(name, KEEP_KEYWORDS):
        return "KEEP", "Defensible for genotype-based baseline comparison (mapping feasible)"

    if keyword_match(name, CONDITIONAL_KEYWORDS):
        return "CONDITIONAL", "Mechanism heterogeneity; requires explicit caveats + coverage reporting"

    # Default: force justification
    return "CONDITIONAL", "Not in curated KEEP list; must justify biologically + methodologically"

# ============================
# Main
# ============================


def main():
    in_path = "data/merged/all_phenotypes_canonical.csv"
    df = pd.read_csv(in_path)

    if "antibiotic" not in df.columns or "phenotype" not in df.columns:
        raise KeyError(
            f"{in_path} must contain columns: antibiotic, phenotype")

    df["antibiotic"] = df["antibiotic"].map(normalize_name)
    df["phenotype"] = df["phenotype"].astype(str).str.strip().str.upper()

    counts = (
        df.groupby("antibiotic")["phenotype"]
          .value_counts()
          .unstack(fill_value=0)
    )

    for col in ["I", "R", "S"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["I", "R", "S"]].copy()

    rows = []
    for abx, row in counts.iterrows():
        decision, reason = classify_antibiotic(abx, row)
        rows.append({
            "antibiotic": abx,
            "I": int(row["I"]),
            "R": int(row["R"]),
            "S": int(row["S"]),
            "total": int(row.sum()),
            "decision": decision,
            "reason": reason,
        })

    out = pd.DataFrame(rows).set_index("antibiotic")
    out = out.sort_values(["decision", "R"], ascending=[True, False])

    print("\n===== SUMMARY =====")
    print(out["decision"].value_counts())

    print("\n===== KEEP =====")
    print(out[out["decision"] == "KEEP"][["I", "R", "S", "total", "reason"]])

    print("\n===== CONDITIONAL =====")
    print(out[out["decision"] == "CONDITIONAL"]
          [["I", "R", "S", "total", "reason"]])

    print("\n===== DROP =====")
    print(out[out["decision"] == "DROP"][["I", "R", "S", "total", "reason"]])

    out_path = "external_validation_antibiotic_screen.csv"
    out.to_csv(out_path)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
