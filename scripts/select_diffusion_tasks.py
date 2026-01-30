#!/usr/bin/env python3
import os, glob, json
import numpy as np

TASK_DIR="ml_tasks"
OUT="results/diffusion_task_selection.json"

# Drop tasks where test has too few examples of either class
MIN_TEST_POS=10
MIN_TEST_NEG=10

# Also drop extreme-prevalence tests (near-constant label)
MIN_TEST_PREV=0.10
MAX_TEST_PREV=0.90

def load(drug, suf):
    return np.load(os.path.join(TASK_DIR, f"{drug}_{suf}.npy"))

drugs=sorted(set(
    os.path.basename(p).replace("_train_y.npy","")
    for p in glob.glob(os.path.join(TASK_DIR,"*_train_y.npy"))
))

keep=[]
drop={}
for d in drugs:
    ytr=load(d,"train_y"); yva=load(d,"val_y"); yte=load(d,"test_y")

    info={
        "train_n": int(len(ytr)), "train_pos": int(ytr.sum()), "train_neg": int(len(ytr)-ytr.sum()),
        "val_n": int(len(yva)),   "val_pos": int(yva.sum()), "val_neg": int(len(yva)-yva.sum()),
        "test_n": int(len(yte)),  "test_pos": int(yte.sum()), "test_neg": int(len(yte)-yte.sum()),
        "test_prev": float(yte.mean()),
    }

    if info["test_pos"] < MIN_TEST_POS or info["test_neg"] < MIN_TEST_NEG:
        drop[d] = {"reason":"too_few_test_examples_per_class", **info}; continue

    if not (MIN_TEST_PREV <= info["test_prev"] <= MAX_TEST_PREV):
        drop[d] = {"reason":"extreme_test_prevalence", **info}; continue

    keep.append({"drug": d, **info})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT,"w") as f:
    json.dump({"keep": keep, "drop": drop}, f, indent=2)

print("KEEP (diffusion eligible):")
for k in keep:
    print(f"  {k['drug']:28s} test_prev={k['test_prev']:.3f}  test_pos={k['test_pos']:3d}  test_neg={k['test_neg']:3d}")
print(f"\nSaved -> {OUT}")
