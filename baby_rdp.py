import pandas as pd
from rdp import rdp
import Baby_washdata_change as wdc
import numpy as np


def calculate_deviation(original_path, simplified_path):
    original_path=original_path.loc['coordinate']
    simplified_path=simplified_path.loc['coordinate']
    deviations = 0
    for point in original_path:
        # 计算每个点到简化路径的最小距离
        min_distance = np.min(np.sqrt(np.sum((simplified_path - point) ** 2, axis=1)))
        deviations=deviations+min_distance
    # 返回平均偏离度
    return deviations

def simplify(x,epsilon):
    data=x['coordinate']
    data_simple=rdp(data,epsilon)
    x['coordinate']=data_simple
    return x

if __name__ == '__main__':
    data_init=wdc.trans_data("washed_data_50_1.csv")
    see=[]
    for i in np.arange(1,0,-0.001):
        epsilon=i
        data=data_init.apply(lambda row: simplify(row,epsilon),axis=1)
        loss_num=data_init.apply(lambda row: calculate_deviation(row,data.loc[row.name]),axis=1)
        loss_num=np.mean(loss_num)
        see.append(loss_num)
    see=pd.DataFrame(see,columns=['See'])
    see.to_excel('see.xlsx')