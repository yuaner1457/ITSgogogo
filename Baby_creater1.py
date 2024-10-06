import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def loss(x1,x2):
    final=np.abs(x1-x2)
    final=np.sum(final,axis=1)#应该按照行进行相加
    return final

def Findpace(D):
    list=[]
    answer=0
    end=D.shape#找到终点的坐标(在迭代过程中会当做目前终点使用)
    x=end[0]
    y=end[1]
    end=(x-1,y-1)
    answer=answer+D[end]
    list.append(end)
    while end != (0,0):
        x=end[0]
        y=end[1]
        D[end]=np.inf
        tem_idx=np.argmin(D[x-1:x+1,y-1:y+1])
        end=(int(x-1+tem_idx/2),y-1+np.mod(tem_idx,2))
        answer = answer + D[end]
        list.append(end)
    return answer,list

#这个函数接受两个输入，sample，std两条时间路径序列，输出是一个浮点数，即其dtw偏差；请注意，两个输入要求为numpy.array
def DTW(sample,std):
    l1=sample.shape[0]
    l2=std.shape[0]
    D=np.full((l1+1,l2+1),np.inf)#D是损失矩阵
    D[0,0]=0
    for i in range(1,l1+1):#进入loss中计算的时候需要减一
        C=loss(sample[i-1],std)
        for j in range(1,l2+1):
            D[i,j]=np.min(D[i-1:i+1,j-1:j+1])+C[j-1]
    lo,path=Findpace(D)
    return lo

def data_processing(a=0.1):#a为筛选系数
    t_step = 5

    def find_exception(df):
        exceptional_id = []
        for flight_id, group in df.groupby('flight_id'):
            t = np.array(group['t'])
            # 生成等差数列
            arithmetic_sequence = np.arange(t[0], t[0] + t_step * len(t), t_step)
            # 计算误差
            errors = t - arithmetic_sequence
            # 计算绝对误差
            mae = np.mean(np.abs(errors))
            if mae / len(t) > a:
                exceptional_id.append(flight_id)
        return exceptional_id

    df = pd.read_csv('train_data_50.csv')
    df['t'] = df['t'].astype(float)
    exception_id = find_exception(df)
    # 清除异常航线
    df = df[~df['flight_id'].isin(exception_id)]
    # 将 x, y, z 列整合成一个向量
    df['coordinate'] = df.apply(lambda row: np.array([row['x'], row['y'], row['z']]), axis=1)
    # 删除原始的 t ,x, y, z 列
    df.drop(columns=['t', 'x', 'y', 'z'], inplace=True)
    df.dropna(inplace=True)
    df = df.groupby('flight_id')['coordinate'].apply(lambda x: np.vstack(x)).reset_index()
    df.set_index('flight_id', inplace=True)
    # 进行Z-score 标准化
    scaler = StandardScaler()
    df['coordinate'] = df['coordinate'].apply(lambda x: scaler.fit_transform(x))
    return df



df = data_processing(0.1)
df.to_csv('washed_data_50_1.csv')

# 假设df是你的数据框，包含flight_id和coordinate列
tracks = df['coordinate'].tolist()  # 将coordinates转换为列表
num_tracks = len(tracks)

# 初始化DTW距离矩阵
dtw_matrix = np.zeros((num_tracks, num_tracks))

# 计算DTW距离
for i in range(num_tracks):
    for j in range(num_tracks):
        dtw_matrix[i, j] = DTW(tracks[i], tracks[j])

#可视化距离矩阵


plt.figure(figsize=(100, 8))
sns.heatmap(dtw_matrix, cmap='viridis')
plt.title('DTW Distance Matrix')
plt.xlabel('Track Index')
plt.ylabel('Track Index')
plt.show()


#聚类分析


# 使用DBSCAN进行聚类
'''这里eps是聚类距离半径最大值，相当于圆的半径，min_samples就是一个类内最小有几个样本'''
dbscan = DBSCAN(metric='precomputed', eps=10000, min_samples=2)
labels = dbscan.fit_predict(dtw_matrix)

# 将聚类结果添加到原数据框中
df['cluster'] = labels




#结果可视化
plt.figure(figsize=(100, 8))
sns.scatterplot(x=range(num_tracks), y=[0] * num_tracks, hue=labels, palette='deep')
plt.title('DBSCAN Clustering Results')
plt.xlabel('Track Index')
plt.ylabel('Cluster')
plt.show()


