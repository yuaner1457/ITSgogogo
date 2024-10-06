import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# 读取已有的距离矩阵
dtw_matrix = np.loadtxt(open("dtw_50.csv", "rb"), delimiter=",", skiprows=0)

# 确定矩阵的行数
num_tracks = dtw_matrix.shape[0]  # 对于对称矩阵，行数等于列数

# 创建 DataFrame
flight_ids = range(1,num_tracks+1)
df = pd.DataFrame(flight_ids, columns=['flight_id'])

#聚类分析

# 使用DBSCAN进行聚类
'''这里eps是聚类距离半径最大值，相当于圆的半径，min_samples就是一个类内最小有几个样本'''
dbscan = DBSCAN(metric='precomputed', eps=10000, min_samples=2)
labels = dbscan.fit_predict(dtw_matrix)

# 将聚类结果添加到原数据框中
df['cluster'] = labels

# 存储ans标注数据大小和eps等聚类方式
df.to_csv('ans_50_eps10000_ms2.csv')



#结果可视化
plt.figure(figsize=(100, 8))
sns.scatterplot(x=range(num_tracks), y=[0] * num_tracks, hue=labels, palette='deep')
plt.title('DBSCAN Clustering Results')
plt.xlabel('Track Index')
plt.ylabel('Cluster')
plt.show()


