# Classical edge prediction matrices.
import networkx as nx


def adamic_adar(G: nx.Graph):
    return list(nx.adamic_adar_index(G))
