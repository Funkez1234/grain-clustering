import pandas as pd
from sklearn.preprocessing import MinMaxScaler

selected_cols = [
    'major_mean','major_std','minor_mean', 'minor_std',
    'aspect_ratio_mean' , 'aspect_ratio_std',
    'contour_area_mean', 'contour_area_std',
    'roughness_mean', 'roughness_std', 
    'solidity_mean', 'solidity_std',
    'mean_red_mean', 'mean_green_mean', 'mean_blue_mean'
]

df = pd.read_csv("aggregated_features_per_image.csv")
X = df[selected_cols].dropna()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(X)

scaled_data_df = pd.DataFrame(scaled_data, columns=selected_cols)
scaled_data_df.to_csv("test_data.csv", index=False)

scaled_data_df = pd.DataFrame(scaled_data, columns=selected_cols)
scaled_data_df.to_csv("scaled_data.csv", index=False)