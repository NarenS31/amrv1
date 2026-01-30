from pathlib import Path
import re
import numpy as np
import pandas as pd

PROB_DIR = Path("results/stronger_v1")
TASK_DIR = Path("ml_tasks")
PHENO_CSV = Path("data/merged/phenotypes_final.csv")
OUT_DIR  = Path("results/stronger_v1")  # keep next to prob files

def parse_prob_filename(name: str):
    """
    Expected patterns:
      <drug>_<model>_<split>_prob.npy
    where <drug> can contain underscores.
    We'll parse from the RIGHT.
    """
    m = re.match(r"^(.*)_(svm|xgb)_(test|val)_prob\.npy$", name)
    if not m:
        return None
    drug, model, split = m.group(1), m.group(2), m.group(3)
    return drug, model, split

def find_label_column(df: pd.DataFrame, drug: str):
    """
    Try common column names.
    You might have:
      antibiotic_norm == drug with phenotype in another column,
    OR one column per drug.
    We handle both possibilities.
    """
    # Case A: wide format (one column per drug)
    if drug in df.columns:
        return drug

    # Case B: long format: rows contain antibiotic name + phenotype
    # Try to detect.
    candidates = [c for c in df.columns if c.lower() in ("antibiotic", "antibiotic_norm", "drug")]
    pheno_cols = [c for c in df.columns if c.lower() in ("phenotype", "pheno", "label", "y", "sir", "r_s")]
    return candidates, pheno_cols

def main():
    prob_files = sorted(PROB_DIR.glob("*_prob.npy"))
    print(f"Found {len(prob_files)} prob files")

    ph = pd.read_csv(PHENO_CSV)

    # Identify accession column (must align with index arrays)
    # Your idx arrays are positions into the dedup combined table you used during task creation.
    # Usually phenotypes_final.csv was aligned row-wise to that table.
    # We will assume row order matches (index-based). If not, you're screwed and need an accession mapping.
    n_rows = len(ph)
    print("phenotypes rows:", n_rows)

    wrote = 0
    skipped = 0

    for pf in prob_files:
        parsed = parse_prob_filename(pf.name)
        if not parsed:
            print("[skip] can't parse:", pf.name)
            skipped += 1
            continue
        drug, model, split = parsed

        # load indices
        idx_path = TASK_DIR / f"{drug}_{split}_idx.npy"
        if not idx_path.exists():
            # some people name them test_idx not test_idx.npy; handle both
            idx_path2 = TASK_DIR / f"{drug}_{split}idx.npy"
            idx_path3 = TASK_DIR / f"{drug}_{split}_indices.npy"
            hits = [p for p in [idx_path2, idx_path3] if p.exists()]
            if hits:
                idx_path = hits[0]
            else:
                print(f"[skip] {drug}: missing {split} idx file (tried {idx_path.name})")
                skipped += 1
                continue

        idx = np.load(idx_path)
        idx = idx.astype(int)
        if idx.max() >= n_rows:
            print(f"[skip] {drug}: idx max {idx.max()} >= phenotypes rows {n_rows} (row-order mismatch)")
            skipped += 1
            continue

        # infer labels
        label_info = find_label_column(ph, drug)
        if isinstance(label_info, str):
            # wide format: column is drug
            col = label_info
            y = ph.loc[idx, col].values
        else:
            # long format
            drug_cols, pheno_cols = label_info
            if not drug_cols or not pheno_cols:
                print(f"[skip] {drug}: couldn't infer long-format columns (need antibiotic_norm + phenotype)")
                skipped += 1
                continue
            drug_col = drug_cols[0]
            pheno_col = pheno_cols[0]

            # filter to this drug then select rows by idx (requires idx to be on the filtered view -> not possible)
            # So long format only works if phenotypes_final.csv is already per-genome per-drug wide.
            print(f"[skip] {drug}: phenotypes_final.csv seems long-format; need wide labels per drug")
            skipped += 1
            continue

        # Convert labels to 0/1 if they are strings
        if y.dtype.kind in ("U", "S", "O"):
            y = np.array([1 if str(v).upper().startswith("R") else 0 for v in y], dtype=int)
        else:
            # numeric; assume already 0/1 or S/I/R encoded
            y = y.astype(float)
            # if it's S/I/R encoded as 0/1/2 etc, collapse to binary R vs not-R
            # common: 1=R, 0=S
            y = np.array([1 if v == 1 else 0 for v in y], dtype=int)

        out = OUT_DIR / f"{drug}_{model}_{split}_y_true.npy"
        np.save(out, y)
        wrote += 1

    print(f"DONE. wrote y_true for {wrote} files; skipped {skipped}")

if __name__ == "__main__":
    main()
