
from sklearn.metrics import silhouette_score
def score(data,label):
    score = silhouette_score(data, label)
    # print("Cluster labels:", y_clusters)
    return score