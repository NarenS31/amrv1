import re
import pathlib

inp = "data/bvbrc/refseq_fna_downloaded_now.txt"
out = "data/bvbrc/assemblies_downloaded_now.txt"

assemblies = []
pat = re.compile(r"(GCF_\d+\.\d+)")
with open(inp) as f:
    for line in f:
        p = pathlib.Path(line.strip())
        m = pat.search(p.name)
        if m:
            assemblies.append(m.group(1))

assemblies = sorted(set(assemblies))
with open(out, "w") as f:
    for a in assemblies:
        f.write(a + "\n")

print(f"[OK] assemblies: {len(assemblies)} -> {out}")
