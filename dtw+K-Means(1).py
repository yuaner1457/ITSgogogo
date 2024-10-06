import numpy as np
import pandas as pd  # 导入 pandas 库
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from fastdtw import fastdtw
from tqdm import tqdm
from rdp import rdp
from joblib import Parallel, delayed


# RDP 航迹简化算法
def rdp_simplify(track, epsilon=5):
    simplified_track = rdp(track, epsilon=epsilon)
    return simplified_track


# 降采样
def downsample_track(track, factor=5):
    return track[::factor]  # 每隔 factor 个点取一个


# 计算每条航迹的欧几里得距离序列（从起点开始）
def compute_euclidean_sequence(track):
    start_point = track[0]
    distances = np.sqrt(np.sum((track - start_point) ** 2, axis=1))
    return distances


# 并行计算 FastDTW 距离矩阵
def calculate_dtw_distance_matrix(tracks):
    num_tracks = len(tracks)
    distance_matrix = np.zeros((num_tracks, num_tracks))

    def compute_dtw(i, j):
        # 对航迹进行降采样并简化
        track_i_simplified = rdp_simplify(downsample_track(tracks[i], factor=5), epsilon=5)
        track_j_simplified = rdp_simplify(downsample_track(tracks[j], factor=5), epsilon=5)
        track_i_sequence = compute_euclidean_sequence(track_i_simplified)
        track_j_sequence = compute_euclidean_sequence(track_j_simplified)
        distance, _ = fastdtw(track_i_sequence, track_j_sequence)
        return i, j, distance

    print("开始并行计算 FastDTW 距离矩阵...")
    # 并行计算
    results = Parallel(n_jobs=-1)(
        delayed(compute_dtw)(i, j) for i in range(num_tracks) for j in range(i + 1, num_tracks))

    # 将结果填入矩阵
    for i, j, distance in results:
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

    print("FastDTW 距离矩阵计算完成。")
    return distance_matrix


# 提取每个flight_id的轨迹
def extract_tracks(data):
    tracks = []
    unique_flight_ids = data['flight_id'].unique()

    print("开始提取每个 flight_id 的轨迹...")
    for flight_id in unique_flight_ids:
        flight_data = data[data['flight_id'] == flight_id]
        track = flight_data[['x', 'y', 'z']].values  # 使用 x, y, z 作为轨迹点
        tracks.append(track)

    print("轨迹提取完成。")
    return tracks, unique_flight_ids


# 使用KMeans进行聚类
def kmeans_clustering(distance_matrix, n_clusters):
    print("开始 KMeans 聚类...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(distance_matrix)
    print("KMeans 聚类完成。")
    return labels


# 处理带有标签的数据
def process_with_labels(file_path):
    print(f"正在读取带有标签的数据：{file_path}...")
    df = pd.read_csv(file_path)
    true_labels = df.groupby('flight_id')['label'].first().values
    print("带有标签的数据读取完成。")
    return true_labels


# 处理无标签的数据并进行聚类
def process_without_labels(file_path, n_clusters=30):
    print(f"正在读取无标签的数据：{file_path}...")
    df = pd.read_csv(file_path)

    print("开始处理无标签的数据...")
    tracks, unique_flight_ids = extract_tracks(df)

    distance_matrix = calculate_dtw_distance_matrix(tracks)

    labels = kmeans_clustering(distance_matrix, n_clusters)

    silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")
    dbi = davies_bouldin_score(distance_matrix, labels)

    unique_clusters = len(np.unique(labels))
    print(f"Number of clusters: {unique_clusters}")
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {dbi}")

    print("无标签数据处理和聚类完成。")
    return labels, silhouette_avg, dbi, unique_flight_ids


# 保存聚类结果到 CSV 文件
def save_labels_to_csv(unique_flight_ids, labels, file_name='predicted_labels.csv'):
    print(f"正在保存聚类结果到 {file_name} 文件...")
    df = pd.DataFrame({'flight_id': unique_flight_ids, 'Cluster Labels': labels})
    df.to_csv(file_name, index=False)
    print(f"聚类结果保存完成，文件名：{file_name}")


# 评估聚类性能
def evaluate_clustering(true_labels, predicted_labels):
    print("开始评估聚类性能...")
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    print(f"Adjusted Rand Index (ARI): {ari}")
    print(f"Normalized Mutual Information (NMI): {nmi}")
    print("聚类性能评估完成。")
    return ari, nmi


if __name__ == "__main__":
    print("程序开始执行...")

    # 处理带有标签的数据，获取真实标签
    true_labels = process_with_labels('valid_data.csv')

    # 处理无标签的数据并进行聚类，获取预测标签以及聚类指标
    predicted_labels, silhouette_avg, dbi, unique_flight_ids = process_without_labels('valid_data_withoutlabels.csv',
                                                                                      n_clusters=30)

    # 保存聚类结果到 CSV 文件
    save_labels_to_csv(unique_flight_ids, predicted_labels)

    # 对聚类结果和真实标签进行评估
    evaluate_clustering(true_labels, predicted_labels)

    print("程序执行完成。")
