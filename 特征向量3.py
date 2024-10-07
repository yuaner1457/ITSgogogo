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
    # 检查输入数据是否包含必要的列
    if set(['t', 'x', 'y', 'z']).issubset(df.columns):
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
    else:
        raise ValueError("输入数据缺少必要的列：'t', 'x', 'y', 'z'")

# 函数：使用匈牙利算法（最优匹配）对聚类标签进行对齐
def align_labels(true_labels, predicted_labels):
    confusion = confusion_matrix(true_labels, predicted_labels)
    row_ind, col_ind = linear_sum_assignment(-confusion)

    # 映射所有标签，确保所有预测标签都能被映射
    label_mapping = {col: row for row, col in zip(col_ind, row_ind)}
    aligned_predicted_labels = [label_mapping.get(label, label) for label in predicted_labels]

    return aligned_predicted_labels

# 加载数据集并进行聚类及分析
try:
    file_path = 'train_data.csv'
    data = pd.read_csv(file_path)

    # 对每条航迹进行重采样
    num_points = 230
    flight_features_resampled = data.groupby('flight_id', group_keys=False).apply(
        lambda df: resample_flight_path(df[['t', 'x', 'y', 'z']], num_points=num_points)
    )

    flight_features_matrix_resampled = np.vstack(flight_features_resampled.values)

    # 标准化特征
    scaler = StandardScaler()
    flight_features_scaled_resampled = scaler.fit_transform(flight_features_matrix_resampled)

    # 层次聚类
    Z = linkage(flight_features_scaled_resampled, method='ward')
    max_d = 75
    flight_clusters_hierarchical = fcluster(Z, max_d, criterion='distance')

    # 保存聚类结果
    flight_features_hierarchical_df = pd.DataFrame({
        'flight_id': flight_features_resampled.index,
        'cluster': flight_clusters_hierarchical
    })

    clustering_result_file = 'clustering_result.csv'
    flight_features_hierarchical_df.to_csv(clustering_result_file, index=False)

    # 合并真实标签
    data_with_labels = pd.read_csv('valid_data.csv')
    merged_data = pd.merge(flight_features_hierarchical_df, data_with_labels[['flight_id', 'label']], on='flight_id')

    true_labels = merged_data['label']
    predicted_clusters = merged_data['cluster']

    # 对齐标签
    aligned_predicted_clusters = align_labels(true_labels, predicted_clusters)

    # 计算 NMI 和 ARI
    nmi_aligned_score = normalized_mutual_info_score(true_labels, aligned_predicted_clusters)
    ari_aligned_score = adjusted_rand_score(true_labels, aligned_predicted_clusters)
    print(f'NMI: {nmi_aligned_score}, ARI: {ari_aligned_score}')

    # 确保 flight_id 唯一，删除重复的行
    merged_data_unique = merged_data.drop_duplicates(subset=['flight_id'])
    aligned_predicted_clusters = merged_data_unique['cluster'].values

    # 评估内部指标
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

except Exception as e:
    print(f"处理过程中发生错误: {e}")



#---------------------
# 确保 flight_id 唯一，删除重复的行
merged_data_unique = merged_data.drop_duplicates(subset=['flight_id'])

# 检查去重后的数据
print(f"去重后的样本数量: {merged_data_unique.shape[0]}")
print(f"特征矩阵的样本数量: {flight_features_scaled_resampled.shape[0]}")

aligned_predicted_clusters = merged_data_unique['cluster'].values

# 检查对齐后的聚类标签数量
print(f"对齐后的聚类标签数量: {len(aligned_predicted_clusters)}")

# 确保特征矩阵和标签数量一致
if len(aligned_predicted_clusters) == flight_features_scaled_resampled.shape[0]:
    # 评估轮廓系数和戴维斯-鲍尔丁指数
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
else:
    print("特征矩阵和标签的样本数量不一致，无法计算内部指标。")



