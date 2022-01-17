import networkx as nx
from sklearn.cluster import spectral_clustering


def run_spectral_clustering(G: nx.Graph):
    # sc = SpectralClustering(
    #    8, affinity="precomputed", n_init=100, assign_labels="discretize"
    # )

    adj_matrix = nx.convert_matrix.to_numpy_array(G)
    # sc.fit_predict(adj_matrix)
    # sc.assign_labels()
    labels = spectral_clustering(
        adj_matrix,
        n_clusters=8,
        n_init=100,
        assign_labels="discretize",
    )
    print(labels.shape)
