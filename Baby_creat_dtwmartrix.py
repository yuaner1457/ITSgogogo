import numpy as np
import pandas as pd
from DTW import DTW
from sklearn.metrics import pairwise_distances
import Baby_washdata_change as wsd
from scipy.spatial.distance import pdist, squareform
from dtaidistance import dtw
from tqdm import tqdm


def main():
    data=wsd.trans_data('simple_path(1).csv')
    flight=data['coordinate']
    flight=flight.tolist()
    lenth=len(flight)
    distance_matrix=np.zeros((lenth,lenth))
    total_combinations = lenth * lenth  # 计算总组合数
    with tqdm(total=total_combinations, desc="Calculating DTW Distance") as pbar:
        for i in range(lenth):
            for j in range(i):
                distance_matrix[i,j]=DTW(flight[i],flight[j])
                pbar.update(1)  # 每完成一个比较更新进度条
    distance_matrix=pd.DataFrame(distance_matrix)
    distance_matrix.to_excel('distance_matrix.xlsx')
    return

if __name__ == '__main__':
    main()