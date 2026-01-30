import pandas as pd

df = pd.read_csv("results/mash/genome_cluster_split.csv")

cluster_col = "cluster_id"
split_col   = "split"

n = len(df)
n_clusters = df[cluster_col].nunique()

counts = df[split_col].value_counts().to_dict()

print("Total genomes after dedup:", n)
print("Clusters:", n_clusters)
print("Split counts:", counts)

cs = df.groupby(cluster_col).size().sort_values(ascending=False)
print("\nTop 10 largest clusters:")
print(cs.head(10).to_string())
print("\nCluster size summary:")
print(cs.describe().to_string())
