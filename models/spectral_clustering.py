from readline import add_history
import networkx as nx
from sklearn.cluster import SpectralClustering


def run_spectral_clustering(G: nx.Graph):
    adj_matrix = nx.convert_matrix.to_numpy_array(G)
    sc = SpectralClustering(2, affinity="precomputed", n_init=100)
    sc.fit(adj_matrix)

    # I think sc.labels_[i] is the the group that node i is in?
    print(sc.labels_)
