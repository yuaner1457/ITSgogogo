import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score, davies_bouldin_score

# 辅助函数：将航迹重采样为固定数量的点（每条航迹作为一个样本）
def resample_flight_path(df, num_points=230):
    if set(['t', 'x', 'y', 'z']).issubset(df.columns):
        t_vals = df['t'].values
        x_vals = df['x'].values
        y_vals = df['y'].values
        z_vals = df['z'].values

        interp_x = interp1d(t_vals, x_vals, kind='linear', fill_value="extrapolate")
        interp_y = interp1d(t_vals, y_vals, kind='linear', fill_value="extrapolate")
        interp_z = interp1d(t_vals, z_vals, kind='linear', fill_value="extrapolate")

        t_new = np.linspace(t_vals.min(), t_vals.max(), num_points)

        resampled_path = np.concatenate([interp_x(t_new), interp_y(t_new), interp_z(t_new)])

        return resampled_path
    else:
        raise ValueError("输入数据缺少必要的列：'t', 'x', 'y', 'z'")

# 加载数据集并进行DBSCAN聚类及分析
try:
    file_path = 'valid_data_withoutlabels.csv'
    data = pd.read_csv(file_path)

    num_points = 230
    flight_features_resampled = data.groupby('flight_id', group_keys=False).apply(
        lambda df: resample_flight_path(df[['t', 'x', 'y', 'z']], num_points=num_points),
        include_groups=False
    )

    flight_features_matrix_resampled = np.vstack(flight_features_resampled.values)

    # 标准化特征
    scaler = StandardScaler()
    flight_features_scaled_resampled = scaler.fit_transform(flight_features_matrix_resampled)

    # 使用DBSCAN进行聚类
    dbscan = DBSCAN(eps=7, min_samples=5)  # 你可以调整 eps 和 min_samples 参数
    flight_clusters_dbscan = dbscan.fit_predict(flight_features_scaled_resampled)

    # 检查是否有噪声点 (-1 表示噪声点)
    num_clusters = len(set(flight_clusters_dbscan)) - (1 if -1 in flight_clusters_dbscan else 0)
    num_noise_points = list(flight_clusters_dbscan).count(-1)
    print(f'共有 {num_clusters} 个簇')
    print(f'检测到 {num_noise_points} 个噪声点')

    # 保存聚类结果
    flight_features_dbscan_df = pd.DataFrame({
        'flight_id': flight_features_resampled.index,
        'cluster': flight_clusters_dbscan
    })

    # 合并真实标签
    data_with_labels = pd.read_csv('valid_data.csv')

    # 确保验证集中的 flight_id 唯一
    data_with_labels_unique = data_with_labels.drop_duplicates(subset=['flight_id'])

    # 对聚类结果和验证集分别按 flight_id 降序排列
    flight_features_dbscan_df_sorted = flight_features_dbscan_df.sort_values(by='flight_id', ascending=False)
    data_with_labels_sorted = data_with_labels_unique.sort_values(by='flight_id', ascending=False)

    # 合并排序后的数据集
    merged_data = pd.merge(flight_features_dbscan_df_sorted, data_with_labels_sorted[['flight_id', 'label']], on='flight_id')

    # 检查合并后的样本数
    print(f"合并后的数据集大小: {merged_data.shape[0]}")
    print(f"聚类特征矩阵的大小: {flight_features_matrix_resampled.shape[0]}")

    # 提取排序后的真实标签和聚类结果
    true_labels = merged_data['label']
    predicted_clusters = merged_data['cluster']

    # 计算外部指标 (NMI 和 ARI)
    nmi_aligned_score = normalized_mutual_info_score(true_labels, predicted_clusters)
    ari_aligned_score = adjusted_rand_score(true_labels, predicted_clusters)
    print(f'NMI: {nmi_aligned_score}, ARI: {ari_aligned_score}')

    # 检查特征矩阵和对齐后的标签数量是否一致
    if len(predicted_clusters) == flight_features_scaled_resampled.shape[0]:
        try:
            silhouette_avg = silhouette_score(flight_features_scaled_resampled, predicted_clusters)
            print(f"轮廓系数 (Silhouette Score): {silhouette_avg}")
        except ValueError as e:
            print(f"计算轮廓系数时出错: {e}")

        try:
            dbi_score = davies_bouldin_score(flight_features_scaled_resampled, predicted_clusters)
            print(f"戴维斯-鲍尔丁指数 (Davies-Bouldin Index): {dbi_score}")
        except ValueError as e:
            print(f"计算戴维斯-鲍尔丁指数时出错: {e}")
    else:
        print("特征矩阵和标签的样本数量不一致，无法计算内部指标。")

    # 将聚类结果输出到文件
    output_file = 'clustered_flight_results_dbscan.csv'
    merged_data.to_csv(output_file, index=False)
    print(f"聚类结果已保存至: {output_file}")

except Exception as e:
    print(f"处理过程中发生错误: {e}")
