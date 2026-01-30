#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

TASK_DIR = "ml_tasks"
OUT_DIR  = "results/cohort_v1"
OUT_CSV  = os.path.join(OUT_DIR, "task_counts_by_drug.csv")

def load_y(path):
    y = np.load(path).astype(int)
    # force 0/1 if any weirdness
    y = (y > 0).astype(int)
    return y

def split_stats(drug, split):
    y_path = os.path.join(TASK_DIR, f"{drug}_{split}_y.npy")
    i_path = os.path.join(TASK_DIR, f"{drug}_{split}_idx.npy")
    if not (os.path.exists(y_path) and os.path.exists(i_path)):
        return None

    y = load_y(y_path)
    idx = np.load(i_path)

    n = int(len(y))
    pos = int(y.sum())
    neg = int(n - pos)
    prev = float(pos / n) if n else float("nan")

    # sanity: idx length should match y length
    idx_n = int(len(idx))
    idx_ok = (idx_n == n)

    return {
        "drug": drug,
        "split": split,
        "n": n,
        "pos": pos,
        "neg": neg,
        "prevalence": prev,
        "idx_n": idx_n,
        "idx_matches_y": bool(idx_ok),
    }

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    drugs = sorted([f[:-5] for f in os.listdir(TASK_DIR) if f.endswith(".json")])

    rows = []
    for drug in drugs:
        for split in ("train","val","test"):
            st = split_stats(drug, split)
            if st is not None:
                rows.append(st)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("❌ No tasks found. Check TASK_DIR=ml_tasks.")

    # Pivot wide for quick judge-friendly view
    wide = df.pivot_table(
        index="drug",
        columns="split",
        values=["n","pos","neg","prevalence","idx_matches_y"],
        aggfunc="first"
    )
    wide.columns = [f"{a}_{b}" for a,b in wide.columns]
    wide = wide.reset_index()

    # Add derived checks
    for split in ("train","val","test"):
        wide[f"degenerate_{split}"] = (wide.get(f"pos_{split}", 0).fillna(0).astype(float) == 0) | \
                                     (wide.get(f"neg_{split}", 0).fillna(0).astype(float) == 0)

    # Sort by test prevalence then by test n
    wide = wide.sort_values(["prevalence_test","n_test"], ascending=[False, False])

    wide.to_csv(OUT_CSV, index=False)
    print(f"✓ wrote {OUT_CSV}")
    print("\nTop 25 by test prevalence:")
    cols_show = ["drug","n_test","pos_test","neg_test","prevalence_test","degenerate_test"]
    print(wide[cols_show].head(25).to_string(index=False))

    bad_idx = wide[(wide.get("idx_matches_y_train", True) == False) |
                   (wide.get("idx_matches_y_val", True) == False) |
                   (wide.get("idx_matches_y_test", True) == False)]
    if len(bad_idx):
        print("\n⚠️ IDX/Y mismatch detected for:")
        print(bad_idx[["drug","idx_matches_y_train","idx_matches_y_val","idx_matches_y_test"]].to_string(index=False))

    deg = wide[wide["degenerate_test"] == True]
    if len(deg):
        print("\n⚠️ Degenerate TEST tasks (all pos or all neg) — these are invalid for AUROC/AUPRC:")
        print(deg[["drug","n_test","pos_test","neg_test","prevalence_test"]].to_string(index=False))

if __name__ == "__main__":
    main()
