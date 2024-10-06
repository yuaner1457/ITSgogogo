import numpy as np
import pandas as pd
from DTW import DTW
from sklearn.metrics import pairwise_distances
import Baby_washdata_change as wsd


def main():
    data=wsd.trans_data('simple_path(1).csv')
    flight=data['coordinate']
    row=flight[0]
    flight = np.nan_to_num(flight)
    distance_matrix = pairwise_distances(flight, metric=DTW)
    distance_matrix=pd.DataFrame(distance_matrix)
    distance_matrix.to_excel('distance_matrix.xlsx')
    return

if __name__ == '__main__':
    main()