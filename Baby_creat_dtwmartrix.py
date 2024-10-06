import numpy as np
import pandas as pd
from DTW import DTW
from sklearn.metrics import pairwise_distances
import Baby_washdata_change as wsd
from scipy.spatial.distance import pdist, squareform



def main():
    data=wsd.trans_data('simple_path(1).csv')
    flight=data['coordinate']
    flight=flight.tolist()
    flight=np.array(flight,dtype=object)
    distance_matrix = pdist(flight,metric=DTW)
    distance_matrix=squareform(distance_matrix)
    distance_matrix=pd.DataFrame(distance_matrix)
    distance_matrix.to_excel('distance_matrix.xlsx')
    return

if __name__ == '__main__':
    main()