import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from joblib import Parallel, delayed

# 计算转向角（用于计算曲率）
def calculate_angle(a, b, c):
    ab = np.array(b) - np.array(a)
    bc = np.array(c) - np.array(b)
    cos_theta = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 限制 cos_theta 在 [-1, 1] 之间
    angle = np.degrees(np.arccos(cos_theta))
    return angle

# 提取特征函数
def extract_features(data):
    unique_flight_ids = data['flight_id'].unique()
    num_flights = len(unique_flight_ids)
    features = np.zeros((num_flights, 13))  # 增加新特征：最大曲率、长度方差、速度特征

    for i, flight_id in enumerate(unique_flight_ids):
        # 获取当前航迹的坐标和时间信息
        flight_data = data[data['flight_id'] == flight_id]
        coords = flight_data[['x', 'y', 'z']].values
        times = flight_data['t'].values

        # 起点和终点
        start_point = coords[0]
        end_point = coords[-1]

        # 轨迹长度
        diff_coords = np.diff(coords, axis=0)  # 计算相邻点之间的差值
        segment_lengths = np.sqrt(np.sum(diff_coords ** 2, axis=1))  # 计算每段的长度
        total_length = np.sum(segment_lengths)  # 总长度
        length_variance = np.var(segment_lengths)  # 长度方差

        # 曲率
        angles = []
        for j in range(1, len(coords) - 1):
            a = coords[j - 1]
            b = coords[j]
            c = coords[j + 1]
            angle = calculate_angle(a, b, c)  # 计算转向角
            angles.append(angle)

        avg_curvature = np.mean(angles) if len(angles) > 0 else 0  # 平均曲率
        max_curvature = np.max(angles) if len(angles) > 0 else 0  # 最大曲率

        # 速度特征
        time_diff = np.diff(times)  # 计算相邻时间点的时间差
        speeds = segment_lengths / time_diff  # 计算速度
        avg_speed = np.mean(speeds) if len(speeds) > 0 else 0  # 平均速度
        max_speed = np.max(speeds) if len(speeds) > 0 else 0  # 最大速度
        speed_variance = np.var(speeds) if len(speeds) > 0 else 0  # 速度方差

        # 存储特征
        features[i, :] = np.concatenate([start_point, end_point, [total_length, length_variance, avg_curvature, max_curvature, avg_speed, max_speed, speed_variance]])

    return features

# 数据预处理函数
def data_processing(file_path):
    df = pd.read_csv(file_path)
    df['t'] = df['t'].astype(float)

    # 删除异常的轨迹
    t_step = 5
    exception_id = []
    for flight_id, group in df.groupby('flight_id'):
        t = group['t'].values
        arithmetic_sequence = np.arange(t[0], t[0] + t_step * len(t), t_step)
        errors = t - arithmetic_sequence
        mae = np.mean(np.abs(errors))
        if mae / len(t) > 0.1:
            exception_id.append(flight_id)
    df = df[~df['flight_id'].isin(exception_id)]

    return df

# 聚类评估
def evaluate_clustering(features, labels):
    unique_labels = np.unique(labels)

    # 如果只有一个簇，跳过评估
    if len(unique_labels) <= 1:
        print("Only one cluster found. Cannot calculate silhouette score or DBI.")
        return

    # 计算 Silhouette Score
    silhouette_avg = silhouette_score(features, labels)
    print(f"Silhouette Score: {silhouette_avg}")

    # 计算戴维斯-鲍尔丁指数
    dbi = davies_bouldin_score(features, labels)
    print(f"Davies-Bouldin Index: {dbi}")

    return silhouette_avg, dbi


if __name__ == "__main__":
    # 数据预处理
    df = data_processing('train_data.csv')

    # 提取特征
    features = extract_features(df)

    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 使用PCA降维
    pca = PCA(n_components=4)  # 降维到4个主成分
    features_reduced = pca.fit_transform(features_scaled)

    # 层次聚类
    # 调整 linkage 方法
    Z = linkage(features_reduced, method='average')  # average' 方法

    # 使用 'distance' criterion 来基于距离阈值提取聚类
    max_distance = 4  # 调整距离阈值
    labels = fcluster(Z, max_distance, criterion='distance')

    # 打印不同簇的数量
    unique_labels = np.unique(labels)
    print(f"Number of clusters: {len(unique_labels)}")

    # 聚类结果评估
    evaluate_clustering(features_reduced, labels)

    # 将 flight_id 和聚类结果保存到文件
    results_df = pd.DataFrame({'flight_id': df['flight_id'].unique(), 'cluster': labels})
    results_df.to_csv('cluster_results.csv', index=False)
    print("聚类结果已保存为 'cluster_results.csv'")
