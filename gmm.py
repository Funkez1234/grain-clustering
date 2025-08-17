import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture


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



# ----------------------------
# GMM Modell trainieren
# ----------------------------
n_components = 5  # Anzahl der Cluster, kann angepasst werden
gmm_full = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
gmm_full.fit(trainings_data)

gmm_tied = GaussianMixture(n_components=n_components, covariance_type='tied', random_state=42)
gmm_tied.fit(trainings_data)

gmm_diag = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=42)
gmm_diag.fit(trainings_data)

gmm_spherical = GaussianMixture(n_components=n_components, covariance_type='spherical', random_state=42)
gmm_spherical.fit(trainings_data)

gmm_full_som = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
gmm_full_som.fit(som_trainings_data)

gmm_tied_som = GaussianMixture(n_components=n_components, covariance_type='tied', random_state=42)
gmm_tied_som.fit(som_trainings_data)

gmm_diag_som = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=42)
gmm_diag_som.fit(som_trainings_data)

gmm_spherical_som = GaussianMixture(n_components=n_components, covariance_type='spherical', random_state=42)
gmm_spherical_som.fit(som_trainings_data)

df = pd.read_csv("validation_data.csv")
validation_data = df[selected_cols].dropna().to_numpy()


df = pd.read_csv("som_validation_data.csv")
som_validation_data = df[selected_cols].to_numpy()

clusterlabels_df = pd.read_csv("clusterlabels.csv")
clusterlabels_df["gmm_full"] = gmm_full.predict(validation_data)
clusterlabels_df["gmm_tied"] = gmm_tied.predict(validation_data)
clusterlabels_df["gmm_diag"] = gmm_diag.predict(validation_data)
clusterlabels_df["gmm_spherical"] = gmm_spherical.predict(validation_data)
clusterlabels_df["gmm_full_som"] = gmm_full_som.predict(som_validation_data)
clusterlabels_df["gmm_tied_som"] = gmm_tied_som.predict(som_validation_data)
clusterlabels_df["gmm_diag_som"] = gmm_diag_som.predict(som_validation_data)
clusterlabels_df["gmm_spherical_som"] = gmm_spherical_som.predict(som_validation_data)
clusterlabels_df.to_csv("clusterlabels.csv", index=False)