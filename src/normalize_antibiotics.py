
import pandas as pd
import re

INP = "data/bvbrc/amr_with_gcf.csv"
OUT = "data/bvbrc/amr_with_gcf_norm.csv"


def norm_ab(x: str) -> str:
    if not isinstance(x, str):
        return x
    x = x.strip().lower()

    # unify separators/spacing
    x = x.replace("\\", "/")
    x = re.sub(r"\s+", " ", x)
    x = x.replace(" / ", "/").replace(" /", "/").replace("/ ", "/")

    # canonicalize common combos
    x = x.replace("piperacillin/tazobactam", "piperacillin-tazobactam")
    x = x.replace("trimethoprim/sulfamethoxazole",
                  "trimethoprim-sulfamethoxazole")
    x = x.replace("amoxicillin/clavulanic acid", "amoxicillin-clavulanate")
    x = x.replace("ampicillin/sulbactam", "ampicillin-sulbactam")
    x = x.replace("ceftazidime/avibactam", "ceftazidime-avibactam")

    # fix obvious garbage typos if present
    x = x.replace("trimethoprim/sulfobactam", "trimethoprim-sulfamethoxazole")

    return x


df = pd.read_csv(INP, low_memory=False)
before = df["Antibiotic"].nunique() if "Antibiotic" in df.columns else None

df["Antibiotic"] = df["Antibiotic"].map(norm_ab)

df.to_csv(OUT, index=False)

after = df["Antibiotic"].nunique()

print(f"[OK] wrote {OUT} rows={len(df)}")
print(f"unique antibiotics: {before} -> {after}")
print("\nTop 25 antibiotics by count:")
print(df["Antibiotic"].value_counts().head(25).to_string())
