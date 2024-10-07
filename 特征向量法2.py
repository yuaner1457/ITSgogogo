import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, confusion_matrix, silhouette_score, davies_bouldin_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# 辅助函数：将航迹重采样为固定数量的点（每条航迹作为一个样本）
def resample_flight_path(df, num_points=230):
    t_vals = df['t'].values
    x_vals = df['x'].values
    y_vals = df['y'].values
    z_vals = df['z'].values

    # 创建用于插值 x, y, z 的函数
    interp_x = interp1d(t_vals, x_vals, kind='linear', fill_value="extrapolate")
    interp_y = interp1d(t_vals, y_vals, kind='linear', fill_value="extrapolate")
    interp_z = interp1d(t_vals, z_vals, kind='linear', fill_value="extrapolate")

    # 生成新的时间值，均匀分布
    t_new = np.linspace(t_vals.min(), t_vals.max(), num_points)

    # 对 x, y, z 进行插值，并将结果展平为一个长的特征向量（每条航迹作为一个样本）
    resampled_path = np.concatenate([interp_x(t_new), interp_y(t_new), interp_z(t_new)])

    return resampled_path

# 函数：使用匈牙利算法（最优匹配）对聚类标签进行对齐
def align_labels(true_labels, predicted_labels):
    # 创建真实标签与预测标签的混淆矩阵
    confusion = confusion_matrix(true_labels, predicted_labels)

    # 执行匈牙利算法（线性分配）来对齐标签
    row_ind, col_ind = linear_sum_assignment(-confusion)

    # 创建预测标签到真实标签的映射
    label_mapping = {col: row for row, col in zip(col_ind, row_ind)}

    # 将预测标签映射为真实标签
    aligned_predicted_labels = [label_mapping.get(label, label) for label in predicted_labels]

    return aligned_predicted_labels

# 函数：评估聚类的内部指标（轮廓系数和戴维斯-鲍尔丁指数）
def evaluate_internal_metrics(features, labels):
    # 计算轮廓系数
    try:
        silhouette_avg = silhouette_score(features, labels)
        print(f'轮廓系数 (Silhouette Score): {silhouette_avg}')
    except ValueError as e:
        print(f"计算轮廓系数时出错: {e}")

    # 计算戴维斯-鲍尔丁指数
    try:
        dbi_score = davies_bouldin_score(features, labels)
        print(f'戴维斯-鲍尔丁指数 (Davies-Bouldin Index): {dbi_score}')
    except ValueError as e:
        print(f"计算戴维斯-鲍尔丁指数时出错: {e}")

# 加载数据集
file_path = 'valid_data_withoutlabels.csv'
data = pd.read_csv(file_path)

# 对每条航迹进行重采样，确保每条航迹是一个样本
num_points = 230
# flight_features_resampled = data.groupby('flight_id', group_keys=False).apply(
#     lambda df: resample_flight_path(df[['t', 'x', 'y', 'z']], num_points=num_points),
#     include_groups=False
# )
flight_features_resampled = data.groupby('flight_id', group_keys=False).apply(
    lambda df: resample_flight_path(df[['t', 'x', 'y', 'z']], num_points=num_points)
)



# 将结果转换为矩阵格式，其中每一行对应一条航迹
flight_features_matrix_resampled = np.vstack(flight_features_resampled.values)

# 标准化特征（确保每条航迹对应一个特征向量）
scaler = StandardScaler()
flight_features_scaled_resampled = scaler.fit_transform(flight_features_matrix_resampled)

# 层次聚类
Z = linkage(flight_features_scaled_resampled, method='ward')
max_d = 75  # 可以调整的阈值，用于控制簇的数量
flight_clusters_hierarchical = fcluster(Z, max_d, criterion='distance')

# 将聚类标签与航迹 ID 关联
flight_features_hierarchical_df = pd.DataFrame({
    'flight_id': flight_features_resampled.index,
    'cluster': flight_clusters_hierarchical
})

# 将聚类结果保存为 CSV 文件
clustering_result_file = 'clustering_result.csv'
flight_features_hierarchical_df.to_csv(clustering_result_file, index=False)

# 加载带有真实标签的数据文件
data_with_labels = pd.read_csv('valid_data.csv')

# 根据 flight_id 合并真实标签和聚类结果
merged_data = pd.merge(flight_features_hierarchical_df, data_with_labels[['flight_id', 'label']], on='flight_id')

# 提取真实标签和预测的聚类结果
true_labels = merged_data['label']
predicted_clusters = merged_data['cluster']

# 对齐预测标签与真实标签
aligned_predicted_clusters = align_labels(true_labels, predicted_clusters)

# 计算归一化互信息（NMI）和调整兰德指数（ARI）
nmi_aligned_score = normalized_mutual_info_score(true_labels, aligned_predicted_clusters)
ari_aligned_score = adjusted_rand_score(true_labels, aligned_predicted_clusters)

print(f'NMI: {nmi_aligned_score}, ARI: {ari_aligned_score}')

# # 评估轮廓系数和戴维斯-鲍尔丁指数
# evaluate_internal_metrics(flight_features_scaled_resampled, aligned_predicted_clusters)
# #----------check
# 确保 flight_id 唯一，删除重复的行
merged_data_unique = merged_data.drop_duplicates(subset=['flight_id'])

# 确保标签和特征数量一致
aligned_predicted_clusters = merged_data_unique['cluster'].values
# print(f"修正后的标签数量: {len(aligned_predicted_clusters)}")
try:
    silhouette_avg = silhouette_score(flight_features_scaled_resampled, aligned_predicted_clusters)
    print(f"轮廓系数 (Silhouette Score): {silhouette_avg}")
except ValueError as e:
    print(f"计算轮廓系数时出错: {e}")

try:
    dbi_score = davies_bouldin_score(flight_features_scaled_resampled, aligned_predicted_clusters)
    print(f"戴维斯-鲍尔丁指数 (Davies-Bouldin Index): {dbi_score}")
except ValueError as e:
    print(f"计算戴维斯-鲍尔丁指数时出错: {e}")


