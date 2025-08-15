import pandas as pd
from sklearn.metrics import adjusted_rand_score
import numpy as np


# Load your data
df = pd.read_csv('clusterlabels.csv')
df["random"] = np.random.randint(0, 5, size=len(df))

# Define the cluster label columns
label_columns = [
    "kmeans", "kmeans2", "kmeans_som",
    "ng","ng_som",
    "art", "art_som",
    "spectral", "spectral_som",
    "gmm_full", "gmm_tied", "gmm_diag", "gmm_spherical",
    "gmm_full_som", "gmm_tied_som", "gmm_diag_som", "gmm_spherical_som",
    "random", 
]

# Create an empty DataFrame for ARI matrix
ari_matrix = pd.DataFrame(index=label_columns, columns=label_columns, dtype=float)

# Fill the matrix
for col1 in label_columns:
    for col2 in label_columns:
        try:
            ari_matrix.loc[col1, col2] = adjusted_rand_score(df[col1], df[col2])
        except Exception as e:
            ari_matrix.loc[col1, col2] = None  # oder np.nan

# Save to CSV
ari_matrix.to_csv('ari_matrix.csv')