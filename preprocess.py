import networkx as nx
import pandas as pd
import numpy as np


def load_celegans():
    pass


def load_main_connectome():
    df = pd.read_csv("data/Connectome_matrix.csv", index_col=0)
    np_arr = df.to_numpy(dtype=np.integer)
    return nx.convert_matrix.from_numpy_array(np_arr)


def load_graph(graph_name: str):
    if graph_name == "karate":
        return nx.karate_club_graph()
    if graph_name == "celegans":
        return load_celegans()
    if graph_name == "connectome":
        return load_main_connectome()
