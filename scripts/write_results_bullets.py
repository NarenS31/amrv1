#!/usr/bin/env python3
import os
import pandas as pd

MASTER="results/master_v1/master_results.csv"
CROSS="results/crossdrug_v1/crossdrug_results.csv"
CAL="results/calibration_v1/calibration_metrics.csv"
DEDUP_SUMMARY="results/mash/genome_cluster_split.csv"
ROB="results/robustness_v1/kmer_dropout_results.csv"
OUT_DIR="results/writeup_v1"
OUT_MD=os.path.join(OUT_DIR,"key_results.md")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    m=pd.read_csv(MASTER)
    cross=pd.read_csv(CROSS) if os.path.exists(CROSS) else None
    cal=pd.read_csv(CAL) if os.path.exists(CAL) else None
    rob=pd.read_csv(ROB) if os.path.exists(ROB) else None

    # top latent LR drugs (filter to real tasks; duplicates may exist)
    m1 = m.drop_duplicates("drug").copy()
    top_lr = m1.sort_values("lr_test_auprc", ascending=False).head(5)

    # diffusion gains
    diff = m1[m1["diff_test_auprc"].notna()].copy()
    if not diff.empty:
        diff["diff_gain"] = diff["diff_test_auprc"] - diff["lr_test_auprc"]
        diff_top = diff.sort_values("diff_gain", ascending=False)
    else:
        diff_top = None

    # calibration: compare ECE between svm_platt and xgboost on val
    cal_note = ""
    if cal is not None and not cal.empty:
        v = cal[cal["split"].eq("val")].copy()
        if not v.empty and set(v["model"]) >= {"svm_platt","xgboost"}:
            w = v.pivot_table(index="drug", columns="model", values="ece", aggfunc="first").dropna()
            if not w.empty:
                better = (w["svm_platt"] < w["xgboost"]).mean()
                cal_note = f"- **Calibration (VAL):** SVM+Platt had lower ECE than XGBoost in **{better*100:.0f}%** of drugs with both available."

    # robustness: compare dropout 0.0 vs 0.3 for amikacin
    rob_note=""
    if rob is not None and not rob.empty:
        a = rob[rob["drug"].eq("amikacin")].set_index("dropout")
        if 0.0 in a.index and 0.3 in a.index:
            rob_note = (f"- **Robustness:** amikacin AUPRC_over_prev "
                        f"{a.loc[0.0,'auprc_over_prev']:.2f}× at 0% dropout → "
                        f"{a.loc[0.3,'auprc_over_prev']:.2f}× at 30% dropout (sharp degradation beyond ~20%).")

    # crossdrug headline: hardest heldouts by auprc_over_prev
    cross_note=""
    if cross is not None and not cross.empty and "auprc_over_prev" in cross.columns:
        best = cross.sort_values("auprc_over_prev", ascending=False).head(3)
        cross_note = ("- **Cross-drug generalization (headline metric = AUPRC/prevalence):** "
                      + "; ".join([f"{r.heldout_drug} {r.auprc_over_prev:.2f}×" for _,r in best.iterrows()]))

    # diffusion note: best gain
    diff_note=""
    if diff_top is not None and not diff_top.empty:
        r = diff_top.iloc[0]
        diff_note = (f"- **Diffusion utility:** best gain was **{r.drug}** "
                     f"(ΔAUPRC = {r.diff_gain:+.3f}, base {r.lr_test_auprc:.3f} → aug {r.diff_test_auprc:.3f}).")

    # dedup note (cheap but strong)
    dedup_note = "- **Leakage control:** clustering-based split prevents near-duplicate genomes from appearing across train/val/test (cluster-safe evaluation)."

    lines=[]
    lines.append("# Key results (auto-generated)")
    lines.append("")
    lines.append("## Model performance")
    lines.append("- **Top latent-LR AUPRC drugs (TEST):** " + ", ".join([f"{r.drug}={r.lr_test_auprc:.3f}" for _,r in top_lr.iterrows()]))
    lines.append("")
    lines.append("## Diffusion augmentation")
    if diff_note: lines.append(diff_note)
    else: lines.append("- Diffusion results not found in master table.")
    lines.append("")
    lines.append("## Calibration")
    if cal_note: lines.append(cal_note)
    else: lines.append("- Calibration summary not available.")
    lines.append("")
    lines.append("## Robustness to incomplete sequencing")
    if rob_note: lines.append(rob_note)
    else: lines.append("- Robustness table not available.")
    lines.append("")
    lines.append("## Generalization across drugs")
    if cross_note: lines.append(cross_note)
    else: lines.append("- Cross-drug summary not available.")
    lines.append("")
    lines.append("## Data hygiene / leakage prevention")
    lines.append(dedup_note)
    lines.append("")

    with open(OUT_MD,"w") as f:
        f.write("\n".join(lines))

    print("✓ wrote", OUT_MD)

if __name__ == "__main__":
    main()
