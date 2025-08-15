import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib




# =======================
#  ART2-Algorithmus
# =======================
class ART2:
    def __init__(self, vigilance=0.8, learning_rate=0.5):
        self.vigilance = vigilance
        self.learning_rate = learning_rate
        self.weights = []

    def _similarity(self, input_vector, weight_vector):
        # Kosinus-Ähnlichkeit
        return np.dot(input_vector, weight_vector) / (np.linalg.norm(input_vector) * np.linalg.norm(weight_vector) + 1e-10)

    def fit(self, X):
        for x in X:
            if not self.weights:
                # erstes Cluster erstellen
                self.weights.append(x.copy())
                continue

            similarities = [self._similarity(x, w) for w in self.weights]
            max_idx = np.argmax(similarities)
            if similarities[max_idx] >= self.vigilance:
                # Gewicht anpassen
                self.weights[max_idx] = self.weights[max_idx] + self.learning_rate * (x - self.weights[max_idx])
            else:
                # Neues Cluster erstellen
                self.weights.append(x.copy())

    def predict(self, X):
        labels = []
        for x in X:
            # Ähnlichkeiten zu allen Clustern berechnen
            similarities = [self._similarity(x, w) for w in self.weights]
            # Index des Clusters mit maximaler Ähnlichkeit
            max_idx = np.argmax(similarities)
            labels.append(max_idx)  # immer existierendes Cluster zurückgeben
        return np.array(labels)

selected_cols = [
    'major_mean','major_std',
    'minor_mean', 'minor_std',
    'aspect_ratio_mean' , 'aspect_ratio_std',
    'contour_area_mean', 'contour_area_std',
    'roughness_mean', 'roughness_std', 
    'solidity_mean', 'solidity_std',
    'mean_red_mean', 'mean_green_mean', 'mean_blue_mean'
]

df = pd.read_csv("scaled_data.csv")
scaled_data = df[selected_cols].dropna().to_numpy()

df = pd.read_csv("som_scaled_data.csv")
som_scaled_data = df[selected_cols].to_numpy()

# =======================
#  Trainieren & Labels
# =======================
art2 = ART2(vigilance=0.85, learning_rate=0.8)
art2.fit(scaled_data)

art2_som = ART2(vigilance=0.925, learning_rate=0.8)
art2_som.fit(som_scaled_data)

df = pd.read_csv("test_data.csv")
test_data = df[selected_cols].dropna().to_numpy()

df = pd.read_csv("som_test_data.csv")
som_test_data = df[selected_cols].to_numpy()

labels = art2.predict(test_data)
labels_som = art2_som.predict(som_test_data)


clusterlabels_df = pd.read_csv("clusterlabels.csv")
clusterlabels_df["art"] = labels
clusterlabels_df["art_som"] = labels_som
clusterlabels_df.to_csv("clusterlabels.csv", index=False)


#print("Cluster-Zuordnung:", labels_1)
print("Anzahl Cluster:", len(art2.weights))

#print("Cluster-Zuordnung:", labels_2)
print("Anzahl Cluster som:", len(art2_som.weights))