from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
import pandas as pd
from tqdm import tqdm  # 用于显示进度条
from concurrent.futures import ThreadPoolExecutor, as_completed

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


# 计算DTW距离
def calculate_dtw_distance_matrix(flight_turning_points):
    flight_ids = list(flight_turning_points.keys())
    num_flights = len(flight_ids)
    distance_matrix = np.zeros((num_flights, num_flights))

    def compute_dtw(i, j):
        points_a = flight_turning_points[flight_ids[i]]
        points_b = flight_turning_points[flight_ids[j]]
        if len(points_a) > 1 and len(points_b) > 1:
            distance, _ = fastdtw(points_a, points_b, dist=euclidean)
        else:
            print(f"Insufficient points for flights {flight_ids[i]} or {flight_ids[j]}, assigning distance as infinity.")
            distance = float('inf')  # 处理不足点的情况

        return i, j, distance

    # 使用多线程计算
    with ThreadPoolExecutor(max_workers=200) as executor:
        futures = {executor.submit(compute_dtw, i, j): (i, j) for i in range(num_flights) for j in
                   range(i + 1, num_flights)}

        for future in tqdm(futures, desc="Calculating DTW distances"):
            i, j = futures[future]
            try:
                i, j, distance = future.result()
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
                print(f"Distance between flight {flight_ids[i]} and flight {flight_ids[j]}: {distance}")
            except Exception as e:
                print(f"Error calculating distance for flight {i} and {j}: {e}")

    return distance_matrix, flight_ids
if __name__ == "__main__":
    file_path = 'valid_data.csv'
    data = pd.read_csv(file_path)
    flight_turning_points = extract_turning_point_features(data)
    distance_matrix, flight_ids = calculate_dtw_distance_matrix(flight_turning_points)
    pd.DataFrame(distance_matrix, index=flight_ids, columns=flight_ids).to_excel('Valid_distance_matrix.xlsx')
