import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

def plot_deduplication_reduction():
    labels = ["Original genomes", "After deduplication"]
    values = [456222, 1325]

    plt.figure(figsize=(7,5))
    plt.bar(labels, values)
    plt.ylabel("Number of genomes")
    plt.title("Strict Deduplication of K. pneumoniae Genomes\n(99.9% ANI Mash clustering)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("figures/deduplication_reduction.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_deduplication_reduction()
