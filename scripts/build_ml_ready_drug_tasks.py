#!/usr/bin/env python3
import os, glob, json
import numpy as np
import pandas as pd

KMER_DIR = "kmers/k31"
DRUG_DIR = "datasets/amr_by_drug"
OUT_DIR  = "ml_tasks"   # writes per-drug y + row indices into X matrices

MIN_POS = 50
MIN_NEG = 50

def load_ids(split):
    with open(os.path.join(KMER_DIR, f"ids_{split}.txt")) as f:
        return [ln.strip() for ln in f if ln.strip()]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # build id -> row index maps
    id2row = {}
    for split in ["train","val","test"]:
        ids = load_ids(split)
        id2row[split] = {gid:i for i,gid in enumerate(ids)}
        print(split, "ids:", len(ids))

    rows = []
    for drug_path in sorted(glob.glob(os.path.join(DRUG_DIR, "*"))):
        if not os.path.isdir(drug_path): 
            continue
        drug = os.path.basename(drug_path)

        files = {s: os.path.join(drug_path, f"{s}.csv") for s in ["train","val","test"]}
        if not all(os.path.exists(p) for p in files.values()):
            continue

        task = {"drug": drug, "splits": {}}
        ok = True

        for split, p in files.items():
            df = pd.read_csv(p, usecols=["genome_id","y"])
            df["genome_id"] = df["genome_id"].astype(str)

            idx = []
            y = []
            missing = 0

            m = id2row[split]
            for gid, yy in zip(df["genome_id"], df["y"]):
                if gid in m:
                    idx.append(m[gid])
                    y.append(int(yy))
                else:
                    missing += 1

            if len(idx) == 0:
                ok = False
                break

            idx = np.array(idx, dtype=np.int32)
            y = np.array(y, dtype=np.int8)

            task["splits"][split] = {
                "n": int(len(y)),
                "pos": int(y.sum()),
                "neg": int(len(y) - y.sum()),
                "missing": int(missing),
                "idx_file": f"{drug}_{split}_idx.npy",
                "y_file": f"{drug}_{split}_y.npy",
            }

            np.save(os.path.join(OUT_DIR, task["splits"][split]["idx_file"]), idx)
            np.save(os.path.join(OUT_DIR, task["splits"][split]["y_file"]), y)

        if not ok:
            continue

        # filter unusable tasks
        if task["splits"]["train"]["pos"] < MIN_POS or task["splits"]["train"]["neg"] < MIN_NEG:
            continue

        with open(os.path.join(OUT_DIR, f"{drug}.json"), "w") as f:
            json.dump(task, f, indent=2)

        rows.append({
            "drug": drug,
            "train_n": task["splits"]["train"]["n"],
            "train_pos": task["splits"]["train"]["pos"],
            "train_neg": task["splits"]["train"]["neg"],
            "val_n": task["splits"]["val"]["n"],
            "test_n": task["splits"]["test"]["n"],
        })

    summary = pd.DataFrame(rows).sort_values("train_n", ascending=False)
    summary.to_csv(os.path.join(OUT_DIR, "tasks_summary.csv"), index=False)
    print("✓ wrote", os.path.join(OUT_DIR, "tasks_summary.csv"))
    print(summary.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
