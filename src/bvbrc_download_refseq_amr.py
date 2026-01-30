#!/usr/bin/env python3
"""
Download BV-BRC genomes + AMR phenotype rows for a target species/taxon.

Requires:
  pip install bvbrc pandas

Outputs:
  data/bvbrc/kp_genomes.csv
  data/bvbrc/kp_amr.csv
"""

from __future__ import annotations
import os
import pandas as pd
import bvbrc as bv

OUT_DIR = "data/bvbrc"
os.makedirs(OUT_DIR, exist_ok=True)

# Klebsiella pneumoniae NCBI taxid is 573 (same number you were using).
TAXID = 573


def to_df(resp) -> pd.DataFrame:
    # bvbrc client returns a response-like object; .json() is typical
    data = resp.json()
    # API sometimes returns a dict with "results"
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    return pd.DataFrame(data)


def main():
    genome = bv.GenomeClient()
    amr = bv.GenomeAmrClient()  # documented in bvbrc API reference

    # 1) Pull genomes for taxon_id
    # NOTE: limit="max" returns up to 25,000 per query per docs.
    g_resp = genome.search(
        genome.taxon_id == TAXID,
        select=[
            genome.genome_id,
            genome.genome_name,
            genome.taxon_id,
            genome.assembly_accession,
            genome.biosample_accession,
            genome.genome_status,
            genome.genome_quality,
        ],
        limit="max",
    )
    gdf = to_df(g_resp)
    gdf.to_csv(os.path.join(OUT_DIR, "kp_genomes.csv"), index=False)
    print("[OK] genomes:", len(gdf), "saved to", f"{OUT_DIR}/kp_genomes.csv")

    # 2) Pull AMR phenotype rows for the same taxon
    # AMR rows are per (genome, antibiotic, phenotype/MIC/etc) depending on what BV-BRC stores.
    a_resp = amr.search(
        amr.taxon_id == TAXID,
        limit="max",
    )
    adf = to_df(a_resp)
    adf.to_csv(os.path.join(OUT_DIR, "kp_amr.csv"), index=False)
    print("[OK] amr rows:", len(adf), "saved to", f"{OUT_DIR}/kp_amr.csv")

    print("\nColumns in AMR file (first 30):")
    print(list(adf.columns)[:30])


if __name__ == "__main__":
    main()
