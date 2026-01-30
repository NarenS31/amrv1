#!/usr/bin/env python3
import argparse
import numpy as np

from sklearn.feature_selection import chi2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmer_matrix", default="data/genomes/kmer_matrix.npy")
    ap.add_argument("--x_index", default="data/ml/X_index.npy")
    ap.add_argument("--y", default="data/ml/Y.npy")
    ap.add_argument("--antibiotics", default="data/ml/antibiotics.txt")
    ap.add_argument("--out_dir", default="data/ml")

    ap.add_argument("--min_var", type=float, default=0.0,
                    help="Variance threshold to drop constant/near-constant features. 0.0 drops constants only.")
    ap.add_argument("--topk_per_ab", type=int, default=5000,
                    help="Top K features by chi2 per antibiotic; union across antibiotics.")
    ap.add_argument("--max_features", type=int, default=20000,
                    help="Cap total selected features after union (keeps most frequent-in-topk).")
    args = ap.parse_args()

    X_full = np.load(args.kmer_matrix, mmap_mode="r")
    X_index = np.load(args.x_index)
    Y = np.load(args.y)
    antibiotics = [l.strip() for l in open(args.antibiotics) if l.strip()]

    X = X_full[X_index].astype(np.float32)  # small enough now (187 x 94826)

    # ----------------------------
    # 1) Variance filter (drop constant features)
    # ----------------------------
    # variance = E[x^2] - (E[x])^2
    mean = X.mean(axis=0)
    mean2 = (X * X).mean(axis=0)
    var = mean2 - mean * mean

    keep_var = var > args.min_var
    Xv = X[:, keep_var]
    kept_feature_ids = np.where(keep_var)[0]

    print("Original X:", X.shape)
    print("After var filter:", Xv.shape)

    # ----------------------------
    # 2) Chi2 top-k per antibiotic (union)
    # ----------------------------
    # We count how often each feature is selected across antibiotics.
    counts = np.zeros(Xv.shape[1], dtype=np.int32)

    for j, ab in enumerate(antibiotics):
        y = Y[:, j]
        mask = y != -1
        yj = y[mask]
        if len(np.unique(yj)) < 2:
            continue

        Xj = Xv[mask]

        # chi2 expects non-negative values (k-mer counts are non-negative)
        scores, _ = chi2(Xj, yj)
        topk = min(args.topk_per_ab, scores.shape[0])
        idx = np.argpartition(scores, -topk)[-topk:]
        counts[idx] += 1

        print(f"[{ab}] labeled={mask.sum()}  selected_topk={topk}")

    # Choose features that appear most often in top-k sets
    # (This helps multi-antibiotic general usefulness)
    if counts.sum() == 0:
        raise RuntimeError(
            "Chi2 selected nothing. Check that X has non-negative values and Y has labels.")

    order = np.argsort(-counts)  # descending
    # keep all that were selected at least once
    selected = order[counts[order] > 0]

    # cap to max_features
    if selected.shape[0] > args.max_features:
        selected = selected[:args.max_features]

    # Map back to original feature indices
    selected_feature_ids = kept_feature_ids[selected]

    X_reduced = X[:, selected_feature_ids]

    out_feat = f"{args.out_dir}/selected_feature_indices.npy"
    out_X = f"{args.out_dir}/X_reduced.npy"

    np.save(out_feat, selected_feature_ids.astype(np.int32))
    np.save(out_X, X_reduced.astype(np.float32))

    print("\n=== FEATURE REDUCTION DONE ===")
    print("Selected features:", selected_feature_ids.shape[0])
    print("Saved:", out_feat)
    print("Saved:", out_X)
    print("X_reduced shape:", X_reduced.shape)


if __name__ == "__main__":
    main()
