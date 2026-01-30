#!/bin/bash
set -euo pipefail

# ================== CONFIG ==================
CONDA="/opt/homebrew/Caskroom/miniforge/base/bin/conda"
ENV_NAME="mash-env"

REF_N="${REF_N:-2000}"          # <-- changed from 5000 to 2000
CHUNK_LINES="${CHUNK_LINES:-10000}"

CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo 8)"
DEFAULT_THREADS="$CORES"
if [ "$DEFAULT_THREADS" -gt 12 ]; then DEFAULT_THREADS=12; fi
THREADS="${MASH_THREADS:-$DEFAULT_THREADS}"

if [ ! -x "$CONDA" ]; then
  echo "ERROR: conda not found at: $CONDA"
  exit 1
fi

MASH_CMD="$CONDA run -n $ENV_NAME mash"
PY_CMD="$CONDA run -n $ENV_NAME python"

$MASH_CMD --version >/dev/null 2>&1 || {
  echo "ERROR: mash not runnable via: $CONDA run -n $ENV_NAME mash"
  exit 127
}

echo "==============================================="
echo "OVERNIGHT MASH DEDUPLICATION PIPELINE"
echo "Started: $(date)"
echo "Env: $ENV_NAME"
echo "REF_N: $REF_N"
echo "Chunk lines: $CHUNK_LINES"
echo "Threads: $THREADS (cores: $CORES)"
echo "==============================================="

mkdir -p results/mash results/mash/chunks

# ================== STEP 1: REF SUBSET ==================
REF_TXT="results/mash/ref${REF_N}_combined.txt"
REF_MSH_PREFIX="results/mash/ref${REF_N}_combined"

if [ ! -f "${REF_MSH_PREFIX}.msh" ]; then
  echo "Step 1: Creating reference subset (${REF_N})..."
  gshuf -n "$REF_N" ALL_GENOMES_COMBINED.txt > "$REF_TXT"
  $MASH_CMD sketch -l "$REF_TXT" -o "$REF_MSH_PREFIX" -p "$THREADS"
  echo "✓ Reference subset created: $(date)"
else
  echo "✓ Skipping reference sketch (exists)"
fi

# ================== STEP 2: SPLIT (VERIFY REAL FILES) ==================
echo "Step 2: Ensuring chunk lists exist..."
LIST_COUNT="$(find results/mash/chunks -maxdepth 1 -type f -name 'genome_list_*' | wc -l | tr -d ' ')"

if [ "$LIST_COUNT" -eq 0 ]; then
  echo "  No genome_list_* found — splitting..."
  rm -f results/mash/chunks/genome_list_*
  split -l "$CHUNK_LINES" ALL_GENOMES_COMBINED.txt results/mash/chunks/genome_list_
  LIST_COUNT="$(find results/mash/chunks -maxdepth 1 -type f -name 'genome_list_*' | wc -l | tr -d ' ')"
  echo "✓ Split complete: ${LIST_COUNT} lists"
else
  echo "✓ Split already present: ${LIST_COUNT} lists"
fi

if [ "$LIST_COUNT" -eq 0 ]; then
  echo "ERROR: split produced zero chunk lists."
  exit 1
fi

# ================== STEP 3: SKETCH CHUNKS ==================
echo "Step 3: Sketching chunks..."
i=0
while IFS= read -r f; do
  i=$((i+1))
  out="results/mash/chunks/all_chunk_${i}"
  if [ -f "${out}.msh" ]; then
    echo "✓ Skipping sketch chunk $i (exists)"
    continue
  fi
  echo "  Sketch chunk $i: $f"
  $MASH_CMD sketch -l "$f" -o "$out" -p "$THREADS"
done < <(find results/mash/chunks -maxdepth 1 -type f -name 'genome_list_*' | sort)

CHUNK_MSH_COUNT="$(find results/mash/chunks -maxdepth 1 -type f -name 'all_chunk_*.msh' | wc -l | tr -d ' ')"
echo "✓ Chunk sketches present: ${CHUNK_MSH_COUNT}"
if [ "$CHUNK_MSH_COUNT" -eq 0 ]; then
  echo "ERROR: zero all_chunk_*.msh created."
  exit 1
fi

# ================== STEP 4: DISTANCES ==================
echo "Step 4: Computing distances..."

# Force recompute if gz exists but empty
if [ -f results/mash/near_dupes_combined_999.tsv.gz ]; then
  NONEMPTY="$(python - << 'PY'
import gzip
p="results/mash/near_dupes_combined_999.tsv.gz"
try:
    with gzip.open(p,"rt") as f:
        for _ in f:
            print(1); break
        else:
            print(0)
except FileNotFoundError:
    print(0)
PY
)"
  if [ "$NONEMPTY" -eq 0 ]; then
    echo "  ⚠️ near_dupes exists but empty — forcing recompute"
    rm -f results/mash/near_dupes_combined_999.tsv.gz
    find results/mash/chunks -maxdepth 1 -name 'dist_done_*.ok' -delete || true
  else
    echo "✓ near_dupes exists and non-empty — skipping distances"
  fi
fi

if [ ! -f results/mash/near_dupes_combined_999.tsv.gz ]; then
  tmp="results/mash/near_dupes_combined_999.tsv"
  : > "$tmp"

  i=0
  while IFS= read -r msh; do
    i=$((i+1))
    marker="results/mash/chunks/dist_done_${i}.ok"
    if [ -f "$marker" ]; then
      echo "✓ Skipping dist chunk $i (already done)"
      continue
    fi

    echo "  Dist chunk $i: $msh"
    $MASH_CMD dist "${REF_MSH_PREFIX}.msh" "$msh" \
      | awk '$3 <= 0.001' >> "$tmp"

    touch "$marker"
  done < <(find results/mash/chunks -maxdepth 1 -type f -name 'all_chunk_*.msh' | sort)

  gzip -c "$tmp" > results/mash/near_dupes_combined_999.tsv.gz
  rm -f "$tmp"
  echo "✓ Distances complete: $(date)"
fi

# ================== STEP 5: DEDUP ==================
echo "Step 5: Deduplicating..."
$PY_CMD scripts/dedup_star_combined.py

echo "==============================================="
echo "FINISHED: $(date)"
echo "==============================================="
