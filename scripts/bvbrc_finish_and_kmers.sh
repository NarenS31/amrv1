#!/usr/bin/env bash
set -euo pipefail

URLS_TSV="data/bvbrc/refseq_download_urls.tsv"
GZ_DIR="data/bvbrc/refseq_fna"
UNZ_DIR="data/bvbrc/refseq_fna_unzipped"
BAD_LIST="data/bvbrc/bad_gz.txt"
NEED_LIST="data/bvbrc/need_download.tsv"

KM_OUT="data/bvbrc/kmer"  # change if you want
PARALLEL_DOWNLOADS="${1:-24}"  # default 24 unless you pass another

mkdir -p "$(dirname "$BAD_LIST")" "$GZ_DIR" "$UNZ_DIR" "$KM_OUT" scripts

echo "=============================="
echo "[A] Download missing .gz files (safe resume)"
echo "=============================="
# Build list of URLs whose output file doesn't exist yet
# URLS_TSV is: GCF...\thttps://.../GCF..._genomic.fna.gz
awk -F'\t' -v gz="$GZ_DIR" '
  {
    url=$2
    n=split(url,a,"/")
    fn=a[n]
    out=gz "/" fn
    # print lines for those not already present
    cmd = "test -f \"" out "\""
    if (system(cmd) != 0) print $0
  }
' "$URLS_TSV" > "$NEED_LIST" || true

need=$(wc -l < "$NEED_LIST" | tr -d ' ')
echo "[INFO] missing downloads: $need"

if [ "$need" -gt 0 ]; then
  # Use xargs with input piped (NOT commandline), so it never hits "too long"
  cut -f2 "$NEED_LIST" | \
    xargs -n 1 -P "$PARALLEL_DOWNLOADS" -I {} bash -c '
      url="{}"
      out="'"$GZ_DIR"'"/"$(basename "$url")"
      tmp="$out.part"
      # skip if already exists (race-safe)
      [ -f "$out" ] && exit 0
      echo "[DL] $(basename "$out")"
      curl -L --retry 12 --retry-delay 2 --fail -o "$tmp" "$url" && mv "$tmp" "$out"
    '
else
  echo "[INFO] no missing downloads"
fi

echo "=============================="
echo "[B] Validate all .gz files and list broken ones"
echo "=============================="
rm -f "$BAD_LIST"
touch "$BAD_LIST"

shopt -s nullglob
for f in "$GZ_DIR"/*.gz; do
  gunzip -t "$f" 2>/dev/null || echo "$f" >> "$BAD_LIST"
done

bad=$(wc -l < "$BAD_LIST" | tr -d ' ')
echo "[INFO] bad gz count: $bad"
if [ "$bad" -gt 0 ]; then
  echo "[INFO] sample bad files:"
  head -n 5 "$BAD_LIST" || true
fi

echo "=============================="
echo "[C] Re-download ONLY broken .gz files"
echo "=============================="
if [ "$bad" -gt 0 ]; then
  while read -r f; do
    name=$(basename "$f")
    # find URL containing this basename
    url=$(awk -F'\t' -v n="$name" '$2 ~ n {print $2; exit}' "$URLS_TSV")
    if [ -z "${url:-}" ]; then
      echo "[WARN] could not find URL for $name (skipping)"
      continue
    fi
    echo "[RE-DL] $name"
    tmp="$f.part"
    rm -f "$tmp"
    curl -L --retry 20 --retry-delay 3 --fail -o "$tmp" "$url" && mv "$tmp" "$f"
  done < "$BAD_LIST"

  echo "[INFO] re-test after re-download..."
  rm -f "$BAD_LIST"
  touch "$BAD_LIST"
  for f in "$GZ_DIR"/*.gz; do
    gunzip -t "$f" 2>/dev/null || echo "$f" >> "$BAD_LIST"
  done
  bad2=$(wc -l < "$BAD_LIST" | tr -d ' ')
  echo "[INFO] bad gz after retry: $bad2"
  if [ "$bad2" -gt 0 ]; then
    echo "[INFO] these are still broken (network/storage issue). you can retry again:"
    head -n 20 "$BAD_LIST" || true
  fi
else
  echo "[INFO] no broken gz files"
fi

echo "=============================="
echo "[D] Unzip (safe resume). Writes .fna into $UNZ_DIR"
echo "=============================="
# unzip everything that doesn't already exist
for f in "$GZ_DIR"/*.gz; do
  base=$(basename "$f" .gz)
  out="$UNZ_DIR/$base"
  [ -f "$out" ] && continue
  # stream unzip; if it fails, delete partial output
  gunzip -c "$f" > "$out" 2>/dev/null || rm -f "$out"
done

echo "[INFO] unzipped count: $(ls "$UNZ_DIR" | wc -l | tr -d " ")"
echo "[INFO] gz count:      $(ls "$GZ_DIR" | wc -l | tr -d " ")"

echo "=============================="
echo "[E] K-mer extraction (BV-BRC unzipped genomes)"
echo "=============================="
# IMPORTANT:
# - your kmer_extraction.py currently scans for .fna
# - unzipped files are ..._genomic.fna (good)
# - if you Ctrl+C, you should not trash progress. best is to write outputs per-genome.
# If your script writes one giant file only at end, you need to modify it. (tell me if so.)

python src/kmer_extraction.py \
  --input_dir "$UNZ_DIR" \
  --out_dir "$KM_OUT"

echo "=============================="
echo "[DONE] All steps completed."
echo "=============================="
