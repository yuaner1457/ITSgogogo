
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt

# Helper function to resample the flight path to a fixed number of points
def resample_flight_path(df, num_points=200):
    t_vals = df['t'].values
    x_vals = df['x'].values
    y_vals = df['y'].values
    z_vals = df['z'].values

    # Create interpolating functions for x, y, z
    interp_x = interp1d(t_vals, x_vals, kind='linear', fill_value="extrapolate")
    interp_y = interp1d(t_vals, y_vals, kind='linear', fill_value="extrapolate")
    interp_z = interp1d(t_vals, z_vals, kind='linear', fill_value="extrapolate")

    # Generate new time values evenly spaced
    t_new = np.linspace(t_vals.min(), t_vals.max(), num_points)

    # Interpolate and flatten the coordinates into one long feature vector
    resampled_path = np.concatenate([interp_x(t_new), interp_y(t_new), interp_z(t_new)])

    return resampled_path

# Load the dataset
file_path = 'valid_data_withoutlabels.csv'
data = pd.read_csv(file_path)

# Resample each flight path to a fixed number of points, excluding the grouping column
num_points = 200
flight_features_resampled = data.groupby('flight_id', group_keys=False).apply(lambda df: resample_flight_path(df[['t', 'x', 'y', 'z']], num_points=num_points))

# Convert the resulting Series into a matrix format suitable for clustering
flight_features_matrix_resampled = np.vstack(flight_features_resampled.values)

# Normalize the features
scaler = StandardScaler()
flight_features_scaled_resampled = scaler.fit_transform(flight_features_matrix_resampled)

# Hierarchical clustering
Z = linkage(flight_features_scaled_resampled, method='ward')
max_d = 30  # This threshold can be adjusted to control the number of clusters
flight_clusters_hierarchical = fcluster(Z, max_d, criterion='distance')

# Attach cluster labels back to flight ids
flight_features_hierarchical_df = pd.DataFrame({
    'flight_id': flight_features_resampled.index,
    'cluster': flight_clusters_hierarchical
})

# Save clustering result to a CSV file
clustering_result_file = 'clustering_result.csv'
flight_features_hierarchical_df.to_csv(clustering_result_file, index=False)

# Load the file with the true labels
data_with_labels = pd.read_csv('valid_data.csv')

# Merge the true labels with the clustering result based on flight_id
merged_data = pd.merge(flight_features_hierarchical_df, data_with_labels[['flight_id', 'label']], on='flight_id')

# Extract the true labels and predicted clusters
true_labels = merged_data['label']
predicted_clusters = merged_data['cluster']

# Compute Normalized Mutual Information (NMI) and Adjusted Rand Index (ARI)
nmi_score = normalized_mutual_info_score(true_labels, predicted_clusters)
ari_score = adjusted_rand_score(true_labels, predicted_clusters)

print(f'NMI: {nmi_score}, ARI: {ari_score}')

# # 检查 'merged_data' 中的 flight_id 对应顺序是否一致
# print(merged_data[['flight_id', 'label', 'cluster']].head())  # 查看前几行数据的 flight_id、label 和 cluster
#
# 比较 flight_id 是否一致
is_matching = merged_data['flight_id'].equals(flight_features_hierarchical_df['flight_id'])
print(f"Flight ID 是否一致: {is_matching}")

# 比较前几个样本的标签
comparison_df = pd.DataFrame({'true_labels': true_labels, 'predicted_clusters': predicted_clusters})
print(comparison_df)  # 打印前几行以便检查是否对齐
#--------------------------------------------------
# 生成带有 flight_id 的 DataFrame
true_labels_df = merged_data[['flight_id', 'label']]
predicted_clusters_df = flight_features_hierarchical_df[['flight_id', 'cluster']]

# 合并两个数据框，确保顺序一致
aligned_df = pd.merge(true_labels_df, predicted_clusters_df, on='flight_id', how='inner')

# 提取对齐后的 true_labels 和 predicted_clusters
true_labels_aligned = aligned_df['label']
predicted_clusters_aligned = aligned_df['cluster']


# 比较 flight_id 是否一致
is_matching = merged_data['flight_id'].equals(flight_features_hierarchical_df['flight_id'])
print(f"Flight ID 是否一致: {is_matching}")

# 重新计算 ARI 和 NMI
nmi_score = normalized_mutual_info_score(true_labels_aligned, predicted_clusters_aligned)
ari_score = adjusted_rand_score(true_labels_aligned, predicted_clusters_aligned)

print(f'NMI: {nmi_score}, ARI: {ari_score}')
