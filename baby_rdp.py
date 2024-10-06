import pandas as pd
from rdp import rdp
import Baby_washdata_change as wdc
import numpy as np


def calculate_deviation(original_path, simplified_path):
    deviations = []
    for point in original_path:
        # 计算每个点到简化路径的最小距离
        min_distance = np.min(np.sqrt(np.sum((simplified_path - point) ** 2, axis=1)))
        deviations.append(min_distance)
    # 返回平均偏离度
    return np.mean(deviations)

def simplify(x,epsilon):
    data=x['coordinate']
    data_simple=rdp(data,epsilon)
    x['coordinate']=data_simple
    return x

if __name__ == '__main__':
    data=wdc.trans_data("washed_data_50_1.csv")
    epsilon=1
    data=data.apply(lambda row: simplify(row,epsilon),axis=1)
    print(data)