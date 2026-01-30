from pathlib import Path
import pandas as pd

AMR_DIR = Path("results/amrfinder_raw")   # or results/frozen/amrfinder_raw
LAB_PATH = Path("results/lab_phenotypes_core.csv")
OUT_JOINED = Path("results/benchmark_joined.csv")
OUT_SUMMARY = Path("results/benchmark_summary.txt")

# 1) Drug -> AMRFinder Class mapping (edit as needed)
DRUG_TO_CLASS = {
    # quinolones
    "ciprofloxacin": "QUINOLONE",
    "levofloxacin": "QUINOLONE",
    "nalidixic acid": "QUINOLONE",
    "norfloxacin": "QUINOLONE",

    # carbapenems
    "meropenem": "CARBAPENEM",
    "imipenem": "CARBAPENEM",

    # aminoglycosides
    "amikacin": "AMINOGLYCOSIDE",
    "gentamicin": "AMINOGLYCOSIDE",
    "tobramycin": "AMINOGLYCOSIDE",
    "netilmicin": "AMINOGLYCOSIDE",
    "streptomycin": "AMINOGLYCOSIDE",

    # tetracyclines
    "tetracycline": "TETRACYCLINE",
    "doxycycline": "TETRACYCLINE",
    "minocycline": "TETRACYCLINE",
    "tigecycline": "TETRACYCLINE",

    # folate pathway antagonists
    "trimethoprim-sulfamethoxazole": "SULFONAMIDE/TRIMETHOPRIM",
    "trimethoprim/sulfamethoxazole": "SULFONAMIDE/TRIMETHOPRIM",
    "trimethoprim": "TRIMETHOPRIM",
    "sulfamethoxazole": "SULFONAMIDE",

    # polymyxins
    "colistin": "COLISTIN",

    # beta-lactams (broad buckets)
    "cefepime": "CEPHALOSPORIN",
    "ceftriaxone": "CEPHALOSPORIN",
    "ceftazidime": "CEPHALOSPORIN",
    "cefazolin": "CEPHALOSPORIN",
    "cefuroxime": "CEPHALOSPORIN",
    "cefotaxime": "CEPHALOSPORIN",
    "cefotetan": "CEPHALOSPORIN",
    "cefoxitin": "CEPHALOSPORIN",

    "aztreonam": "MONOBACTAM",  # fixed typo (was "MONOB ACTAM")
    "ampicillin": "PENICILLIN",
    "ampicillin-sulbactam": "PENICILLIN",
    "amoxicillin-clavulanate": "PENICILLIN",
    "piperacillin-tazobactam": "PENICILLIN",
    "ticarcillin/clavulanic acid": "PENICILLIN",

    # other
    "nitrofurantoin": "NITROFURANTOIN",
    "chloramphenicol": "PHENICOL",
    "florfenicol": "PHENICOL",
    "fosfomycin": "FOSFOMYCIN",
}


def normalize_drug(x: str) -> str:
    x = str(x).strip().lower()
    x = x.replace("_", "-")
    x = x.replace("\\", "/")
    return x


def gcf_base(gcf: str) -> str:
    # "GCF_000406765.2" -> "GCF_000406765"
    g = str(gcf).strip()
    return g.split(".", 1)[0]


def load_amrfinder_hits(amr_dir: Path) -> pd.DataFrame:
    rows = []
    for tsv in sorted(amr_dir.glob("*.tsv")):
        try:
            df = pd.read_csv(tsv, sep="\t", dtype=str)
        except Exception:
            continue
        if df.empty or "Name" not in df.columns:
            continue

        # Genome id is in Name column (because you used --name)
        # Fallback: filename stem
        name0 = df["Name"].iloc[0] if pd.notna(
            df["Name"].iloc[0]) else tsv.stem
        genome = str(name0).strip()

        for _, r in df.iterrows():
            rows.append({
                "gcf": genome,
                "gcf_base": gcf_base(genome),
                "element_symbol": r.get("Element symbol"),
                "element_name": r.get("Element name"),
                "type": r.get("Type"),
                "class": r.get("Class"),
                "subclass": r.get("Subclass"),
                "method": r.get("Method"),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["class"] = out["class"].fillna("").str.strip()
    out["subclass"] = out["subclass"].fillna("").str.strip()
    return out


def build_pred_table(lab: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    lab = lab.copy()
    lab["gcf_base"] = lab["gcf"].map(gcf_base)

    lab["antibiotic_norm"] = lab["antibiotic"].map(normalize_drug)
    lab["amr_class"] = lab["antibiotic_norm"].map(DRUG_TO_CLASS)

    # collapse amrfinder hits to set of classes present per genome (JOIN ON BASE)
    genome_to_classes = (
        hits.groupby("gcf_base")["class"]
        .apply(lambda s: set([x for x in s if x]))
        .to_dict()
    )

    def predict(row) -> str:
        if pd.isna(row["amr_class"]):
            return "UNK"
        classes = genome_to_classes.get(row["gcf_base"], set())
        return "R" if row["amr_class"] in classes else "S"

    lab["amrfinder_pred"] = lab.apply(predict, axis=1)
    return lab


def score(joined: pd.DataFrame) -> str:
    df = joined.copy()
    df = df[df["amrfinder_pred"].isin(["S", "R"])]
    df = df[df["p"].isin(["S", "R"])]

    if df.empty:
        return "No scorable rows after filtering."

    tp = ((df["amrfinder_pred"] == "R") & (df["p"] == "R")).sum()
    tn = ((df["amrfinder_pred"] == "S") & (df["p"] == "S")).sum()
    fp = ((df["amrfinder_pred"] == "R") & (df["p"] == "S")).sum()
    fn = ((df["amrfinder_pred"] == "S") & (df["p"] == "R")).sum()

    acc = (tp + tn) / (tp + tn + fp + fn)

    return "\n".join([
        f"Rows scored: {len(df)}",
        f"TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}",
        f"Accuracy: {acc:.4f}",
        f"Precision (R): {tp/(tp+fp):.4f}" if (tp+fp) else "Precision (R): NA",
        f"Recall (R): {tp/(tp+fn):.4f}" if (tp+fn) else "Recall (R): NA",
    ])


def main():
    lab = pd.read_csv(LAB_PATH, dtype=str)

    # Accept either schema:
    if "gcf" not in lab.columns and "genome_id" in lab.columns:
        lab = lab.rename(columns={"genome_id": "gcf"})
    if "p" not in lab.columns and "phenotype" in lab.columns:
        lab = lab.rename(columns={"phenotype": "p"})

    need = {"gcf", "antibiotic", "p"}
    missing = need - set(lab.columns)
    if missing:
        raise SystemExit(
            f"Lab file missing columns: {missing}. Have: {list(lab.columns)}")

    hits = load_amrfinder_hits(AMR_DIR)
    if hits.empty:
        raise SystemExit(f"No AMRFinder hits loaded from: {AMR_DIR}")

    joined = build_pred_table(lab, hits)
    joined.to_csv(OUT_JOINED, index=False)

    summary = score(joined)
    OUT_SUMMARY.write_text(summary + "\n")
    print(summary)
    print(f"\nWrote: {OUT_JOINED}")
    print(f"Wrote: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
