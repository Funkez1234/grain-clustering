import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib


selected_cols = [
'major_mean','major_std','minor_mean', 'minor_std',
'aspect_ratio_mean' , 'aspect_ratio_std',
'contour_area_mean', 'contour_area_std',
'roughness_mean', 'roughness_std', 
'solidity_mean', 'solidity_std',
'mean_red_mean', 'mean_green_mean', 'mean_blue_mean'
]

def plot_clusters(X_scaled, cluster_labels, centroids, axis_features, title="3D Scatter mit NG"):

    X_scaled = pd.DataFrame(X_scaled, columns=selected_cols)
    centroids = pd.DataFrame(centroids, columns=selected_cols)

    x, y, z = axis_features
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster_id in enumerate(unique_clusters):
        cluster_points = X_scaled[cluster_labels == cluster_id]
        ax.scatter(
            cluster_points[x], cluster_points[y], cluster_points[z],
            color=colors[i],
            label=f'Cluster {cluster_id}',
            s=10, alpha=0.7
        )

    # Zentren plotten
    ax.scatter(
        centroids[x], centroids[y], centroids[z],
        marker='*', c='black', s=200, label='Zentroid'
    )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title(title)
    ax.legend()


class NeuralGas:
    def __init__(self, n_neurons=10, max_iter=1000, epsilon_initial=0.5, epsilon_final=0.005, lambda_initial=10.0, lambda_final=0.5):
        self.n_neurons = n_neurons
        self.max_iter = max_iter
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.lambda_initial = lambda_initial
        self.lambda_final = lambda_final

    def fit(self, X):
        n_samples, n_features = X.shape
        np.random.seed(42)  # <--- Set before initializing weights
        self.weights = np.random.rand(self.n_neurons, n_features)

        for t in range(self.max_iter):
            # Dynamische Parameter
            eps_t = self.epsilon_initial * (self.epsilon_final / self.epsilon_initial) ** (t / self.max_iter)
            lambda_t = self.lambda_initial * (self.lambda_final / self.lambda_initial) ** (t / self.max_iter)

            # Shuffle der Daten
            for x in X[np.random.permutation(n_samples)]:
                # Distanzen & Rangliste
                distances = np.linalg.norm(self.weights - x, axis=1)
                ranks = np.argsort(np.argsort(distances))  # Rang 0 = bester

                # Update für jede Neuronengewichtung
                for i in range(self.n_neurons):
                    h = np.exp(-ranks[i] / lambda_t)
                    self.weights[i] += eps_t * h * (x - self.weights[i])

        return self

    def predict(self, X):
        # Weise jedem Punkt das nächste Neuron zu (Cluster-ID)
        distances = np.linalg.norm(X[:, np.newaxis] - self.weights, axis=2)
        return np.argmin(distances, axis=1)


df = pd.read_csv("trainings_data.csv")
trainings_data = df[selected_cols].dropna().to_numpy()

df = pd.read_csv("som_trainings_data.csv")
som_trainings_data = df[selected_cols].to_numpy()

ng = NeuralGas(n_neurons=5, max_iter=500)
ng.fit(trainings_data)
ng_som = NeuralGas(n_neurons=5, max_iter=500)
ng_som.fit(som_trainings_data)



df = pd.read_csv("validation_data.csv")
validation_data = df[selected_cols].dropna().to_numpy()

df = pd.read_csv("som_validation_data.csv")
som_validation_data = df[selected_cols].to_numpy()

labels = ng.predict(trainings_data)
labels_som = ng_som.predict(som_validation_data)


clusterlabels_df = pd.read_csv("clusterlabels.csv")
clusterlabels_df["ng"] = labels
clusterlabels_df["ng_som"] = labels_som
clusterlabels_df.to_csv("clusterlabels.csv", index=False)
    


axis_features = ['roughness_mean', 'minor_std', 'contour_area_mean']  
plot_clusters(trainings_data, labels, ng.weights, axis_features)
#plt.show()