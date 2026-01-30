#!/usr/bin/env bash
set -euo pipefail

IN_LIST="results/amrfinder_test/batch_test.txt"
OUT_DIR="results/amrfinder_test"
mkdir -p "$OUT_DIR"

# If amrfinder isn't on PATH, this will fail immediately (good).
command -v amrfinder >/dev/null 2>&1 || { echo "amrfinder not found on PATH"; exit 1; }

n=0
while IFS= read -r fa; do
  # skip blanks and comments
  [[ -z "${fa}" ]] && continue
  [[ "${fa:0:1}" == "#" ]] && continue

  if [[ ! -f "$fa" ]]; then
    echo "[WARN] missing fasta: $fa"
    continue
  fi

  base="$(basename "$fa")"
  id="${base%.*}"  # strip extension (e.g., .fna/.fa)

  out_tsv="${OUT_DIR}/${id}.tsv"
  out_err="${OUT_DIR}/${id}.stderr.txt"

  # run
  amrfinder --plus --organism Klebsiella_pneumoniae --threads 4 -n "$fa" -o "$out_tsv" 2> "$out_err"

  n=$((n+1))
  if (( n % 200 == 0 )); then
    echo "done: $n"
  fi
done < "$IN_LIST"

echo "✓ finished AMRFinder on test set. genomes processed: $n"
