import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import joblib


selected_cols = [
'major_mean','major_std','minor_mean', 'minor_std',
'aspect_ratio_mean' , 'aspect_ratio_std',
'contour_area_mean', 'contour_area_std',
'roughness_mean', 'roughness_std', 
'solidity_mean', 'solidity_std',
'mean_red_mean', 'mean_green_mean', 'mean_blue_mean'
]

def plot_kmeans_clusters(scaled_data, cluster_labels, centroids, axis_features, title="3D Scatter mit KMeans"):


    idx_x = selected_cols.index(axis_features[0])
    idx_y = selected_cols.index(axis_features[1])
    idx_z = selected_cols.index(axis_features[2])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster_id in enumerate(unique_clusters):
            cluster_points = scaled_data[cluster_labels == cluster_id]
            ax.scatter(
                cluster_points[:, idx_x], cluster_points[:, idx_y], cluster_points[:, idx_z],
                color=colors[i],
                label=f'Cluster {cluster_id}',
                s=10, alpha=0.7
            )

    # Zentren plotten
    ax.scatter(
        centroids[:, idx_x], centroids[:, idx_y], centroids[:, idx_z],
        marker='*', c='black', s=200, label='Zentroid'
    )

    ax.set_xlabel(axis_features[0])
    ax.set_ylabel(axis_features[1])
    ax.set_zlabel(axis_features[2])
    ax.set_title(title)
    ax.legend()


def elbow_method(X_scaled):
    inertias = []
    k_range = range(1, 15)

    for k in k_range:
        scores = []
        for seed in range(10):  # 10 verschiedene random_state-Werte
            kmeans = KMeans(n_clusters=k, random_state=seed)
            kmeans.fit(X_scaled)
            scores.append(kmeans.inertia_)
        avg_inertia = np.mean(scores)
        inertias.append(avg_inertia)

    plt.figure()
    plt.plot(k_range, inertias, marker='o')
    plt.xlabel("Anzahl der Cluster (k)")
    plt.ylabel("Inertia (Summe der Abstände)")
    plt.title("Elbow-Methode zur Bestimmung von k")
    plt.grid(True)
 

def silhouette_scores(X_scaled):

    silhouette_scores = []
    k_range = range(2, 16)  # Beispiel: k von 2 bis 10

    for k in k_range:
        scores = []
        for seed in range(10):  # 10 verschiedene random states
            kmeans = KMeans(n_clusters=k, random_state=seed)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            scores.append(score)
        avg_score = np.mean(scores)
        silhouette_scores.append(avg_score)

    plt.figure()
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.xlabel("Anzahl der Cluster (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette-Methode")
    plt.grid(True)


scaled_data_df = pd.read_csv("scaled_data.csv")
scaled_data = scaled_data_df[selected_cols].dropna().to_numpy()

som_scaled_data_df = pd.read_csv("som_scaled_data.csv")
som_scaled_data = som_scaled_data_df[selected_cols].to_numpy()

#Bestimmung von Clusteranzahl
elbow_method(scaled_data)
silhouette_scores(scaled_data)


k = 5  # Anzahl Cluster ermittelt über silhouette_scores
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(scaled_data)
centroids = kmeans.cluster_centers_

kmeans2 = KMeans(n_clusters=k, random_state=40)
kmeans2.fit(scaled_data)
centroids2 = kmeans2.cluster_centers_

kmeans_som = KMeans(n_clusters=k, random_state=42)
kmeans_som.fit(som_scaled_data)
centroids_som = kmeans.cluster_centers_



test_data_df = pd.read_csv("test_data.csv")
test_data = test_data_df[selected_cols].dropna().to_numpy()

som_test_data_df = pd.read_csv("som_test_data.csv")
som_test_data = som_test_data_df[selected_cols].to_numpy()


distances = cdist(test_data, centroids)  # shape: (n_test, n_clusters)
labels = np.argmin(distances, axis=1)

distances2 = cdist(test_data, centroids2)  # shape: (n_test, n_clusters)
labels2 = np.argmin(distances2, axis=1)

distances_som = cdist(som_test_data, centroids_som)  # shape: (n_test, n_clusters)
labels_som = np.argmin(distances_som, axis=1)

clusterlabels_df = pd.read_csv("clusterlabels.csv")
clusterlabels_df["kmeans"] = labels
clusterlabels_df["kmeans2"] = labels2
clusterlabels_df["kmeans_som"] = labels_som
clusterlabels_df.to_csv("clusterlabels.csv", index=False)

axis_features = ['major_mean', 'minor_mean', 'mean_green_mean']  
plot_kmeans_clusters(scaled_data, kmeans.labels_, centroids, axis_features)
plt.show()
