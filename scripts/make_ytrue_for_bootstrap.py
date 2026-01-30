#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import json
import re

PHENO = "data/merged/phenotypes_final.csv"   # or phenotypes_clean.csv if that's what you used
TASK_DIR = Path("ml_tasks")
OUT_DIR = Path("results/stronger_v1")

def norm_drug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def load_task(drug_norm: str):
    # try common task file patterns
    candidates = [
        TASK_DIR / f"{drug_norm}.json",
        TASK_DIR / f"{drug_norm}_task.json",
        TASK_DIR / f"{drug_norm}_ml_task.json",
    ]
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text())
    # fallback: scan for any json containing this drug name
    for p in TASK_DIR.glob("*.json"):
        try:
            t = json.loads(p.read_text())
            d = t.get("drug") or t.get("antibiotic") or t.get("antibiotic_norm")
            if d and norm_drug(d) == drug_norm:
                return t
        except Exception:
            pass
    return None

def main():
    ph = pd.read_csv(PHENO)

    # expected columns
    if "antibiotic_norm" in ph.columns:
        ph["drug_norm"] = ph["antibiotic_norm"].astype(str).map(norm_drug)
    elif "antibiotic" in ph.columns:
        ph["drug_norm"] = ph["antibiotic"].astype(str).map(norm_drug)
    else:
        raise SystemExit("Phenotypes file missing antibiotic column")

    if "phenotype" not in ph.columns:
        raise SystemExit("Phenotypes file missing phenotype column (expected 'phenotype')")

    # map phenotype to 0/1 (R=1, S=0). If you have I, treat as 1? (choose)
    def pheno_to_y(x):
        x = str(x).strip().upper()
        if x == "R":
            return 1
        if x == "S":
            return 0
        # treat Intermediate as resistant by default (common in clinical ML), change if you want
        if x == "I":
            return 1
        return np.nan

    ph["y"] = ph["phenotype"].map(pheno_to_y)
    ph = ph.dropna(subset=["y"])
    ph["y"] = ph["y"].astype(int)

    prob_files = sorted(OUT_DIR.glob("*prob*.npy"))
    if not prob_files:
        raise SystemExit(f"No prob files found in {OUT_DIR}. You need saved test probabilities per drug.")

    print(f"Found {len(prob_files)} prob files")

    wrote = 0
    for pf in prob_files:
        name = pf.stem  # filename without .npy
        # try to extract drug_norm from filename
        # e.g. "ciprofloxacin_xgb_prob" -> "ciprofloxacin"
        parts = name.split("_")
        drug_norm = parts[0]
        # sometimes it’s like "ciprofloxacin_prob" already
        if drug_norm in ("svm","xgb","logreg","lr","platt"):
            # weird naming; skip unless you paste your directory listing
            continue

        task = load_task(drug_norm)
        if task is None:
            print(f"[skip] can't find task json for {drug_norm} (from {pf.name})")
            continue

        # task should contain test indices or test rows. Common keys:
        test_idx = None
        for k in ("test_idx", "idx_test", "test_indices"):
            if k in task:
                test_idx = task[k]
                break

        # alternatively, some pipelines store explicit test rows as genome_id list
        test_genomes = None
        for k in ("test_genomes","genomes_test","test_ids"):
            if k in task:
                test_genomes = task[k]
                break

        # build y_true for the test set
        if test_genomes is not None:
            df = ph[ph["drug_norm"] == drug_norm]
            # if phenotype file has genome_id, join on it
            if "genome_id" not in df.columns:
                print(f"[skip] phenotypes file has no genome_id column for {drug_norm}")
                continue
            df = df.set_index("genome_id")
            y_true = []
            missing = 0
            for gid in test_genomes:
                if gid in df.index:
                    y_true.append(int(df.loc[gid, "y"]))
                else:
                    missing += 1
            if missing:
                print(f"[warn] {drug_norm}: missing {missing} genome labels in phenotypes")
            y_true = np.array(y_true, dtype=int)
        elif test_idx is not None:
            # Here we assume phenotypes rows are aligned to some master index used for tasks.
            # Most robust: if task stores row indices into phenotypes file, use them directly.
            # We'll try: task has 'test_rows' that index into pheno dataframe.
            # If 'test_idx' are kmer rows indices, we cannot map without genome_ids list.
            # We'll attempt to detect by range.
            test_idx = list(map(int, test_idx))
            # If task contains phenotype_row_idx, use it:
            for k in ("test_pheno_rows","test_row_idx","test_rows"):
                if k in task:
                    test_idx = list(map(int, task[k]))
                    break
            if max(test_idx) < len(ph):
                y_true = ph.iloc[test_idx]["y"].to_numpy(dtype=int)
            else:
                print(f"[skip] {drug_norm}: test indices exceed phenotype rows; need genome_id mapping")
                continue
        else:
            print(f"[skip] {drug_norm}: task has no test idx or genomes list")
            continue

        yprob = np.load(pf)
        if len(yprob) != len(y_true):
            print(f"[skip] {drug_norm}: len(prob)={len(yprob)} != len(y_true)={len(y_true)} (alignment issue)")
            continue

        out_y = OUT_DIR / f"{drug_norm}_y_true.npy"
        np.save(out_y, y_true)
        wrote += 1
        print(f"[ok] {drug_norm}: wrote {out_y.name} (n={len(y_true)})")

    print(f"\nDONE. wrote y_true for {wrote} drugs")

if __name__ == "__main__":
    main()
