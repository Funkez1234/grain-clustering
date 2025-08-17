import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom



selected_cols = [
'major_mean','major_std','minor_mean', 'minor_std',
'aspect_ratio_mean' , 'aspect_ratio_std',
'contour_area_mean', 'contour_area_std',
'roughness_mean', 'roughness_std', 
'solidity_mean', 'solidity_std',
'mean_red_mean', 'mean_green_mean', 'mean_blue_mean'
]

df = pd.read_csv("trainings_data.csv")
scaled_data = df[selected_cols].dropna().to_numpy()    # Nur ausgewählte Features(selected_cols)

# Normalisieren
RANDOM_SEED=42
scaler = MinMaxScaler()

som_x, som_y = 10, 10  # Grid-Größe
som = MiniSom(x=som_x, y=som_y, input_len=scaled_data.shape[1], sigma=1, learning_rate=0.75,random_seed=RANDOM_SEED)
som.random_weights_init(scaled_data)
som.train_random(data=scaled_data, num_iteration=1000)
weights = som.get_weights().reshape(som_x * som_y, -1)


# ---SOM Cluster Map (U-Matrix)---
plt.figure(figsize=(10, 8))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # U-Matrix
plt.colorbar(label='Distanz (U-Matrix)')
plt.title("SOM Cluster Map (U-Matrix)")
plt.grid()


# ---Leere SOM-Neuronen Map (U-Matrix)---
#zeigt neuronen die nicht benutzt werden bei 10x10 werden aber alle benutzt
plt.figure(figsize=(10, 8))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # Hintergrund: U-Matrix
plt.colorbar(label='Distanz (U-Matrix)')
all_neurons = [(x, y) for x in range(som_x) for y in range(som_y)]
used_neurons = set(som.win_map(scaled_data).keys())
unused_neurons = [pos for pos in all_neurons if pos not in used_neurons]
for w in unused_neurons:
    plt.plot(w[0] + 0.5 , w[1] + 0.5, 'x', color='red', markersize=10, markeredgewidth=2)
plt.title("Leere SOM-Neuronen")
plt.grid()


   
# ---SOM RGB Color Map---
rgb_features = ['major_mean', 'minor_mean', 'contour_area_mean']
rgb_indices = [selected_cols.index(f) for f in rgb_features]
rgb_map = np.zeros((som_x, som_y, 3))  # Initialize RGB map
for x in range(som_x):
    for y in range(som_y):
        neuron_weights = som.get_weights()[x, y]
        rgb = neuron_weights[rgb_indices]  # Extract R,G,B values
        rgb_map[x, y] = np.clip(rgb, 0, 1)  # Ensure values are in [0,1] range
plt.figure(figsize=(10, 8))
plt.imshow(rgb_map, origin='lower')
plt.title("SOM RGB Color Map (based on neuron weights)")
plt.xticks(range(som_x))
plt.yticks(range(som_y))
plt.grid(False)

#plt.show()


test_df = pd.read_csv("validation_data.csv")
test_data = test_df[selected_cols].dropna()

som_df = pd.DataFrame(
    weights,
    columns=selected_cols
)
som_df.to_csv("som_trainings_data.csv", index=False)


bmu_coordinates = []
for x in test_data.to_numpy():
    bmu = som.winner(x)  # (x, y) coordinate
    bmu_coordinates.append(bmu)
        
bmu_weights = []
for coord in bmu_coordinates:
    x, y = coord
    bmu_weights.append(som._weights[x, y])  # weight vector for BMU

coords_df = pd.DataFrame({
    "BMU_X": [coord[0] for coord in bmu_coordinates],
    "BMU_Y": [coord[1] for coord in bmu_coordinates],
})
weights_df = pd.DataFrame(bmu_weights, columns=[selected_cols[i] for i in range(len(bmu_weights[0]))])
som_df = pd.concat([coords_df.reset_index(drop=True), weights_df], axis=1)
som_df.to_csv("som_validation_data.csv", index=False)

