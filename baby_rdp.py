import numpy as np
import pandas as pd
from rdp import rdp
import ast

def clean(row,axis):

        # 解析字符串为列表
    result =  np.fromstring(row,sep=' ')
        # 转换为 NumPy 数组
    result = np.array(result)
    return result
def calculate_deviation(original_path, simplified_path):
    deviations = []
    for point in original_path:
        # 计算每个点到简化路径的最小距离
        min_distance = np.min(np.sqrt(np.sum((simplified_path - point) ** 2, axis=1)))
        deviations.append(min_distance)
    # 返回平均偏离度
    return np.mean(deviations)

def simplify(x):
    data=x.to_numpy()
    data_simple=rdp(data,1)
    print(data_simple)
    return calculate_deviation(data,data_simple)

if __name__ == '__main__':
    data=pd.read_csv('washed_data_50_1.csv')
    data=data['coordinate']
    data=data.apply(clean,axis=1)
    test=ast.literal_eval('[[1,2,3],[4,6,5]]')
    x=simplify(data)
    print(x)