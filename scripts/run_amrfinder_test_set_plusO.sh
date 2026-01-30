#!/usr/bin/env bash
set -euo pipefail

IN_LIST="results/amrfinder_test/batch_test.txt"
OUT_DIR="results/amrfinder_test_plusO"
ORGANISM="${1:-Klebsiella_pneumoniae}"

mkdir -p "$OUT_DIR"

i=0
while IFS= read -r fa; do
  [[ -z "$fa" ]] && continue
  [[ "$fa" =~ ^# ]] && continue

  base="$(basename "$fa")"
  gid="${base%.fna}"
  out_tsv="$OUT_DIR/$gid.tsv"
  out_err="$OUT_DIR/$gid.stderr.txt"

  [[ -s "$out_tsv" ]] && continue

  # --plus: use AMRFinderPlus db; -O: organism mutation searches
  ( mamba run -n amrfinder amrfinder --plus -O "$ORGANISM" -n "$fa" -o "$out_tsv" ) 2> "$out_err" || true

  i=$((i+1))
  if (( i % 200 == 0 )); then
    echo "done: $i"
  fi
done < "$IN_LIST"

echo "✓ finished -> $OUT_DIR"
