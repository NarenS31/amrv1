import numpy as np
from collections import Counter
from tqdm import tqdm

INPUT_PATH = "data/genomes/kmer_features.npy"
OUTPUT_PATH = "data/genomes/kmer_matrix.npy"


def main():
    print("Loading k-mer dictionaries...")
    kmer_dicts = np.load(INPUT_PATH, allow_pickle=True)
    print(f"Loaded {len(kmer_dicts)} genomes")

    print("Building global k-mer vocabulary...")
    vocab = Counter()
    for d in tqdm(kmer_dicts):
        vocab.update(d.keys())

    kmer_list = sorted(vocab.keys())
    kmer_to_idx = {k: i for i, k in enumerate(kmer_list)}
    print(f"Total unique k-mers: {len(kmer_list)}")

    X = np.zeros((len(kmer_dicts), len(kmer_list)), dtype=np.float32)
    print("Converting dicts to matrix...")
    for i, d in enumerate(tqdm(kmer_dicts)):
        for kmer, count in d.items():
            X[i, kmer_to_idx[kmer]] = count

    np.save(OUTPUT_PATH, X)
    print(f"Saved matrix to {OUTPUT_PATH}")
    print("Final shape:", X.shape)


if __name__ == "__main__":
    main()
