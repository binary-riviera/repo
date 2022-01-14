# Classical edge prediction matrices.
import networkx as nx


def adamic_adar(G: nx.Graph):
    return list(nx.adamic_adar_index(G))


def jaccard_coefficient(G: nx.Graph):
    return list(nx.jaccard_coefficient(G))


def preferential_attachment(G: nx.Graph):
    return list(nx.preferential_attachment(G))
