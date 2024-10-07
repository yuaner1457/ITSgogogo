from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
import pandas as pd
from tqdm import tqdm  # 用于显示进度条


# 检测航迹中的转折点
def detect_turning_points(coords):
    turning_points = []
    for j in range(1, len(coords) - 1):
        a = coords[j - 1]
        b = coords[j]
        c = coords[j + 1]
        angle = calculate_angle(a, b, c)
        if angle < 150:  # 小于150度认为是一个转折点，阈值可以调整
            turning_points.append(b)
    return turning_points


# 计算夹角
def calculate_angle(a, b, c):
    ab = np.array(b) - np.array(a)
    bc = np.array(c) - np.array(b)
    cos_theta = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 防止超出范围
    angle = np.degrees(np.arccos(cos_theta))
    return angle


# 提取航迹的转折点特征
def extract_turning_point_features(data):
    unique_flight_ids = data['flight_id'].unique()
    flight_turning_points = {}

    for flight_id in tqdm(unique_flight_ids, desc="Extracting turning points"):
        flight_data = data[data['flight_id'] == flight_id]
        coords = flight_data[['x', 'y', 'z']].values
        turning_points = detect_turning_points(coords)
        flight_turning_points[flight_id] = turning_points

    return flight_turning_points


# 使用 DTW 计算航迹之间的距离
def calculate_dtw_distance_matrix(flight_turning_points):
    flight_ids = list(flight_turning_points.keys())
    num_flights = len(flight_ids)
    distance_matrix = np.zeros((num_flights, num_flights))

    for i in tqdm(range(num_flights), desc="Calculating DTW distances"):
        for j in range(i + 1, num_flights):
            points_a = flight_turning_points[flight_ids[i]]
            points_b = flight_turning_points[flight_ids[j]]

            if len(points_a) > 1 and len(points_b) > 1:  # 确保至少有两个转折点
                distance, _ = fastdtw(points_a, points_b, dist=euclidean)
            else:
                # 如果转折点不足，给定一个较大的距离，避免影响聚类结果
                distance = float('inf')

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix, flight_ids


# 聚类并进行量化评估
def evaluate_clustering(distance_matrix, flight_ids, true_labels=None):
    # 使用层次聚类
    Z = linkage(distance_matrix, method='ward')

    # 设置聚类阈值，根据距离进行聚类
    labels = fcluster(Z, 5, criterion='maxclust')

    # 评估聚类效果（量化指标）
    silhouette_avg = silhouette_score(distance_matrix, labels, metric='precomputed')
    dbi = davies_bouldin_score(distance_matrix, labels)

    # 打印聚类评估指标
    print(f"Number of clusters: {len(np.unique(labels))}")
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {dbi}")

    # 如果提供了真实标签，计算 ARI 和 NMI
    if true_labels is not None:
        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels)
        print(f"Adjusted Rand Index (ARI): {ari}")
        print(f"Normalized Mutual Information (NMI): {nmi}")
        return labels, silhouette_avg, dbi, ari, nmi

    return labels, silhouette_avg, dbi


# 保存聚类结果到 CSV 文件
def save_cluster_results(flight_ids, labels, output_path='cluster_results.csv'):
    cluster_results = pd.DataFrame({
        'flight_id': flight_ids,
        'cluster': labels
    })
    cluster_results.to_csv(output_path, index=False)
    print(f"Cluster results saved to {output_path}")


# 主函数入口
if __name__ == "__main__":
    # 读取数据
    file_path = 'short_valid_data_withoutlabels.csv'
    true_labels_file = 'short_valid_data.csv'  # 如果有真实标签文件

    data = pd.read_csv(file_path)

    # 提取转折点特征并计算 DTW 距离矩阵
    flight_turning_points = extract_turning_point_features(data)
    distance_matrix, flight_ids = calculate_dtw_distance_matrix(flight_turning_points)

    # 加载真实标签数据
    true_labels_data = pd.read_csv(true_labels_file)
    true_labels = true_labels_data.groupby('flight_id')['label'].first().values

    # 聚类并评估结果
    labels, silhouette_avg, dbi, ari, nmi = evaluate_clustering(distance_matrix, flight_ids, true_labels=true_labels)

    # 保存聚类结果到 CSV 文件
    save_cluster_results(flight_ids, labels, output_path='cluster_results.csv')
