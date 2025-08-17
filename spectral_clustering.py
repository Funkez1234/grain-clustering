from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib


# Ausgewählte Features
selected_cols = [
    'major_mean','major_std','minor_mean', 'minor_std',
    'aspect_ratio_mean' , 'aspect_ratio_std',
    'contour_area_mean', 'contour_area_std',
    'roughness_mean', 'roughness_std', 
    'solidity_mean', 'solidity_std',
    'mean_red_mean', 'mean_green_mean', 'mean_blue_mean'
]

df = pd.read_csv("trainings_data.csv")
trainings_data = df[selected_cols].dropna().to_numpy()

df = pd.read_csv("som_trainings_data.csv")
som_trainings_data = df[selected_cols].to_numpy()

df = pd.read_csv("validation_data.csv")
validation_data = df[selected_cols].dropna().to_numpy()

df = pd.read_csv("som_validation_data.csv")
som_validation_data = df[selected_cols].to_numpy()

# Anzahl der Cluster festlegen
n_clusters = 5  # Beispiel: 3 Cluster

# Spectral Clustering-Objekt erstellen
spectral1 = SpectralClustering(
    n_clusters=n_clusters,
    affinity='rbf',  # Radial Basis Function (Gaussian) für die Ähnlichkeit
    gamma=1.0,       # Parameter für RBF
    assign_labels='kmeans',
    random_state=42
)
spectral2 = SpectralClustering(
    n_clusters=n_clusters,
    affinity='rbf',  # Radial Basis Function (Gaussian) für die Ähnlichkeit
    gamma=1.0,       # Parameter für RBF
    assign_labels='kmeans',
    random_state=42
)

cluster_labels = spectral1.fit_predict(trainings_data)
cluster_labels = spectral2.fit_predict(som_trainings_data)

clusterlabels_df = pd.read_csv("clusterlabels.csv")
clusterlabels_df["spectral"] = spectral1.fit_predict(validation_data)
clusterlabels_df["spectral_som"] = spectral2.fit_predict(som_validation_data)
clusterlabels_df.to_csv("clusterlabels.csv", index=False)
