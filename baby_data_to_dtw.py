import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


df = pd.read_csv('washed_data_500_1.csv')

# 假设df是你的数据框，包含flight_id和coordinate列
tracks = df['coordinate'].tolist()  # 将coordinates转换为列表
num_tracks = len(tracks)

#这里有严重报错 如果先将处理好的数据存好 在计算DTW的这个函数再读入就会编程字符串 哪位gg能在这里加个函数处理一下QAQ

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


np.savetxt("dtw_500.csv", dtw_matrix, delimiter=',')
