import numpy as np
import pandas as pd
from DTW import DTW
from sklearn.metrics import pairwise_distances
import Baby_washdata_change as wsd
from scipy.spatial.distance import pdist, squareform
from dtaidistance import dtw



def main():
    data=wsd.trans_data('simple_path(1).csv')
    flight=data['coordinate']
    flight=flight.tolist()
    lenth=len(flight)
    distance_matrix=np.zeros((lenth,lenth))
    for i in range(lenth):
        for j in range(lenth):
            distance_matrix[i,j]=DTW(flight[i],flight[j])
        print(f'{i}/{lenth}')
    distance_matrix=pd.DataFrame(distance_matrix)
    distance_matrix.to_excel('distance_matrix.xlsx')
    return

if __name__ == '__main__':
    main()