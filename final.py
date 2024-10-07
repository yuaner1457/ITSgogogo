import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score, davies_bouldin_score


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


# 加载数据集并进行聚类及分析
try:
    file_path = 'valid_data.csv'
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

    # 使用 PCA 进行降维，保留 95% 的方差
    pca = PCA(n_components=0.95)
    flight_features_pca = pca.fit_transform(flight_features_scaled_resampled)

    # 打开文件写入输出信息
    with open('输出说明.txt', 'w', encoding='utf-8') as output_file:

        # 写入PCA后的信息
        # output_file.write(f"PCA降维后保留的主成分数量: {flight_features_pca.shape[1]}\n")

        # 层次聚类
        Z = linkage(flight_features_pca, method='ward')
        max_d = 100
        flight_clusters_hierarchical = fcluster(Z, max_d, criterion='distance')

        # 保存聚类结果
        flight_features_hierarchical_df = pd.DataFrame({
            'flight_id': flight_features_resampled.index,
            'cluster': flight_clusters_hierarchical
        })

        # 计算内部指标
        num_clusters = len(np.unique(flight_clusters_hierarchical))
        print(f'共有 {num_clusters} 个簇')
        output_file.write(f'共有 {num_clusters} 个簇\n')

        try:
            silhouette_avg = silhouette_score(flight_features_pca, flight_clusters_hierarchical)
            print(f"轮廓系数 (Silhouette Score): {silhouette_avg}")
            output_file.write(f"轮廓系数 (Silhouette Score): {silhouette_avg}\n")
        except ValueError as e:
            print(f"计算轮廓系数时出错: {e}")
            output_file.write(f"计算轮廓系数时出错: {e}\n")

        try:
            dbi_score = davies_bouldin_score(flight_features_pca, flight_clusters_hierarchical)
            print(f"戴维斯-鲍尔丁指数 (Davies-Bouldin Index): {dbi_score}")
            output_file.write(f"戴维斯-鲍尔丁指数 (Davies-Bouldin Index): {dbi_score}\n")
        except ValueError as e:
            print(f"计算戴维斯-鲍尔丁指数时出错: {e}")
            output_file.write(f"计算戴维斯-鲍尔丁指数时出错: {e}\n")

    # 将聚类结果输出到文件（不包含标签）
    clustered_output_file = 'clustered_flight_results_pca.csv'
    flight_features_hierarchical_df.to_csv(clustered_output_file, index=False)
    print(f"聚类结果已保存至: {clustered_output_file}")

    with open('输出说明.txt', 'a', encoding='utf-8') as output_file:
        output_file.write(f"聚类结果已保存至: {clustered_output_file}\n")

except Exception as e:
    print(f"处理过程中发生错误: {e}")
    with open('输出说明.txt', 'a', encoding='utf-8') as output_file:
        output_file.write(f"处理过程中发生错误: {e}\n")
