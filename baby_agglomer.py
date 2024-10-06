from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from baby_to_dtw import DTW
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import numpy as np
from sklearn.metrics import silhouette_score




def main(X,distance_matrix,i):
    # 创建层次聚类模型
    model = AgglomerativeClustering(
        n_clusters=i,                # 目标簇数量
        metric='precomputed',        # 使用自定义距离函数
        linkage='single',              # 使用Ward链接
        distance_threshold=None       # 不使用距离阈值
    )
    # 拟合模型并获取聚类标签
    y_clusters = model.fit_predict(distance_matrix)
    score = silhouette_score(X, y_clusters)
    # print("Cluster labels:", y_clusters)
    return score

if __name__ == '__main__':
    loss=np.zeros(1000)
    X=pd.read_csv('/root/.ipython/washed_data.csv',header=None)
    # 尝试将所有数据转换为浮点数
    X=X.to_numpy()
    X=np.delete(X,0,axis=0)
    distance_matrix = squareform(pdist(X, metric=DTW))
    distance_matrix.tofile('distance_matrix.csv')
    print('数据准备完成')
    for i in range(1000):
        loss_num=main(X,distance_matrix,i+1)
        loss[i]=loss_num
        print(f'执行进度{i+1}/1000')
    loss=pd.DataFrame(loss)
    loss.to_csv('agglomer_loss_1to1000.csv')
    print('完成')