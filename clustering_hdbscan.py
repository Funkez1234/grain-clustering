import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import hdbscan
import umap.umap_ as umap
import matplotlib.pyplot as plt
import numpy as np

selected_cols = [
'major_mean','major_std','minor_mean', 'minor_std',
'aspect_ratio_mean' , 'aspect_ratio_std',
'contour_area_mean', 'contour_area_std',
'roughness_mean', 'roughness_std', 
'solidity_mean', 'solidity_std',
'mean_red_mean', 'mean_green_mean', 'mean_blue_mean'
]

# CSV laden
df = pd.read_csv("aggregated_features_per_image.csv")

# Nur ausgewählte Features und NaNs entfernen
X = df[selected_cols].dropna()

# Normalisieren
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Dimension mit UMAP auf 2D reduzieren (nur für Plot)
umap_2d = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_2d.fit_transform(X_scaled)

# HDBSCAN auf den Originaldaten anwenden
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean')
labels = clusterer.fit_predict(X_scaled)

# Ergebnisstatistik
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_ratio = np.mean(labels == -1)
print(f"Gefundene Cluster (ohne Rauschen): {n_clusters}")
print(f"Anteil Rauschen: {noise_ratio:.2%}")

# Plot mit UMAP-Koordinaten
plt.figure(figsize=(8, 6))
palette = plt.cm.get_cmap('tab10', np.max(labels) + 2)

for label in np.unique(labels):
    mask = (labels == label)
    color = 'k' if label == -1 else palette(label)
    plt.scatter(X_umap[mask, 0], X_umap[mask, 1], s=50, c=[color], label=f'Cluster {label}')

plt.title("HDBSCAN Clustering (UMAP Visualisierung)")
plt.legend()
plt.show()
