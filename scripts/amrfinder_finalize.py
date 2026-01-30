#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path

def pick_col(df, candidates=(), contains=()):
    cols = list(df.columns)
    low = {c: c.lower() for c in cols}

    # exact/near-exact candidates first
    for want in candidates:
        for c in cols:
            if low[c] == want.lower():
                return c

    # then substring contains matches
    for sub in contains:
        hits = [c for c in cols if sub.lower() in low[c]]
        if len(hits) == 1:
            return hits[0]
        # if multiple, prefer ones that look like test metrics
        if hits:
            pref = [c for c in hits if "test" in low[c]]
            if len(pref) == 1:
                return pref[0]
            # else prefer plain auprc/auroc without extra junk
            short = sorted(hits, key=lambda x: len(x))
            return short[0]
    return None

def normalize_master(master):
    # drug column
    drug_col = pick_col(master, candidates=("drug","antibiotic","antibiotic_norm"), contains=("drug","antibiotic"))
    if not drug_col:
        raise SystemExit(f"Can't find drug column in master_results. Columns: {list(master.columns)}")

    # model column
    model_col = pick_col(master, candidates=("model","clf","classifier"), contains=("model","clf","classifier","method"))
    if not model_col:
        # some tables store model name under "repr_model" or similar
        model_col = pick_col(master, contains=("svm","xgb","logreg","lr"))
    if not model_col:
        raise SystemExit("Can't find model column (try adding one, or rename your column to 'model').")

    # auprc/auroc columns (often test_auprc/test_auroc)
    auprc_col = pick_col(master, candidates=("auprc","test_auprc"), contains=("auprc",))
    auroc_col = pick_col(master, candidates=("auroc","test_auroc"), contains=("auroc",))

    if not auprc_col or not auroc_col:
        raise SystemExit(
            f"Can't find AUROC/AUPRC columns.\n"
            f"Found auprc_col={auprc_col}, auroc_col={auroc_col}\n"
            f"Columns: {list(master.columns)}"
        )

    # prevalence column (optional but useful)
    prev_col = pick_col(master, candidates=("prev","prevalence","test_prev"), contains=("prev","preval",))

    # ratio column (optional)
    ratio_col = pick_col(master, candidates=("auprc_over_prev","auprc_over_prevalence","ratio"), contains=("over_prev","over_preval","ratio"))

    out = master.copy()
    out = out.rename(columns={
        drug_col: "drug",
        model_col: "model",
        auprc_col: "auprc",
        auroc_col: "auroc",
    })
    if prev_col:
        out = out.rename(columns={prev_col: "prev"})
    else:
        out["prev"] = pd.NA

    if ratio_col:
        out = out.rename(columns={ratio_col: "auprc_over_prev"})
    else:
        out["auprc_over_prev"] = pd.NA

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fair_metrics", default="results/amrfinder/fair_metrics.csv")
    ap.add_argument("--ml_master", default="results/master_v1/master_results.csv")
    ap.add_argument("--out_dir", default="results/final_tables")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fm = pd.read_csv(args.fair_metrics)
    fm0 = fm.copy()
    fm = fm.drop_duplicates(subset=["drug"], keep="first").sort_values("drug")
    (out_dir / "amrfinder_fair_metrics_dedup.csv").write_text(fm.to_csv(index=False))

    headline = fm[["drug","amrfinder_auprc","amrfinder_auroc","test_n","test_prev","amrfinder_pred_mean","missing_tsvs_in_task"]].copy()
    headline.to_csv(out_dir / "amrfinder_headline.csv", index=False)

    master_raw = pd.read_csv(args.ml_master)
    master = normalize_master(master_raw)

    # choose best ML row per drug
    def pick_best(df):
        if df["auprc_over_prev"].notna().any():
            return df.sort_values(["auprc_over_prev","auprc","auroc"], ascending=False).iloc[0]
        return df.sort_values(["auprc","auroc"], ascending=False).iloc[0]

    best = master.groupby("drug", as_index=False).apply(pick_best).reset_index(drop=True)

    merged = headline.merge(best[["drug","model","auroc","auprc","prev","auprc_over_prev"]],
                            on="drug", how="left")

    merged["amrfinder_auprc_over_prev"] = merged["amrfinder_auprc"] / merged["test_prev"].replace(0, pd.NA)
    merged["ml_auprc_over_prev"] = merged["auprc_over_prev"]
    merged["delta_auprc"] = merged["auprc"] - merged["amrfinder_auprc"]
    merged["delta_auroc"] = merged["auroc"] - merged["amrfinder_auroc"]

    merged = merged.sort_values(["ml_auprc_over_prev","delta_auprc"], ascending=False)
    merged.to_csv(out_dir / "amrfinder_vs_best_ml.csv", index=False)

    print("✓ wrote", out_dir / "amrfinder_fair_metrics_dedup.csv")
    print("✓ wrote", out_dir / "amrfinder_headline.csv")
    print("✓ wrote", out_dir / "amrfinder_vs_best_ml.csv")
    print(f"note: fair_metrics rows before={len(fm0)} after_dedup={len(fm)}")

if __name__ == "__main__":
    main()
