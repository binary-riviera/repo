import numpy as np
import networkx as nx
import pandas as pd
from dgl.data import DGLDataset
import random
import matplotlib.pyplot as plt
from torch import clamp


def generate_observed_graph(G: nx.Graph, n_samples: int):
    # print(f"Sampling {n_samples} edges from graph...")
    sampled_edges = random.sample(G.edges, n_samples)
    G_observed = G.copy()
    G_observed.remove_edges_from(sampled_edges)
    # print(G_observed.number_of_nodes())
    return (G_observed, sampled_edges)


def generate_observed_graph_connected(
    G: nx.Graph, n_samples: int, max_component_limit=1
):
    new_G = G.copy()
    sampled_edges = []
    while n_samples > 0:
        edge_to_remove = random.choice(list(new_G.edges))
        new_G.remove_edge(*edge_to_remove)
        if type(new_G) is nx.DiGraph:
            num_cc = len(list(nx.weakly_connected_components(G)))
        else:
            num_cc = nx.number_connected_components(G)

        if num_cc <= max_component_limit:
            sampled_edges.append(edge_to_remove)
            n_samples -= 1
        else:
            new_G.add_edge(*edge_to_remove)
            continue
    return (new_G, sampled_edges)


class Dataset:
    def generate_splits(self, start_p=0.05, end_p=0.45, step=0.05, keep_connected=True):
        self.splits = []  # tuple (Graph, removed_edges)
        for x in np.arange(start_p, end_p, step):
            print(f"removing {x} nodes from graph g")
            if keep_connected:
                self.splits.append(
                    generate_observed_graph_connected(
                        self.G, int(self.G.number_of_edges() * x), max_component_limit=1
                    )
                )
            else:
                self.splits.append(
                    generate_observed_graph(self.G, int(self.G.number_of_edges() * x))
                )


class Connectome(Dataset):
    def __init__(self):
        df = pd.read_csv("data/Connectome_matrix.csv", index_col=0)
        np_arr = df.to_numpy(dtype=np.integer)
        temp_G = nx.convert_matrix.from_numpy_array(np_arr, create_using=nx.DiGraph)

        largest_cc = max(nx.weakly_connected_components(temp_G), key=len)
        self.G = temp_G.subgraph(largest_cc).copy()


class AutonomousSystems(Dataset):
    def __init__(self):
        with open("data/as20graph.txt", "rb") as as_file:
            self.G = nx.read_edgelist(as_file, comments="#")
            print(self.G)


class Facebook(Dataset):
    def __init__(self):
        with open("data/facebook_combined.txt", "rb") as fb_file:
            self.G = nx.read_edgelist(fb_file, comments="#")
            print(self.G)


def write_edge_test_split(G: nx.Graph, sampled, train_name: str, test_name: str):

    with open(train_name, "w") as train_file:
        # we want to write G here
        nodes = G.number_of_nodes()
        for i in range(nodes):
            for j in range(nodes):
                if G.has_edge(i, j) and i != j:
                    # write the edge to file
                    train_file.write(f"{i} {j}\n")

    with open(test_name, "w") as test_file:
        # we want to write sampled here
        for (u, v) in sampled:
            test_file.write(f"{u} {v}\n")


def print_graph_metrics(G: nx.Graph):
    print("Graph Metrics\n")
    print(f"# nodes: {G.number_of_nodes()}")
    print(f"# edges: {G.number_of_edges()}")
    print(f"Weighted?")  # TODO: implement
    print(f"# Components: {len(list(nx.connected_components(G)))}")
    print(f"Average Shortest Path: {nx.average_shortest_path_length(G)}")
    print(f"Average Clustering Coefficient: {nx.average_clustering(G)}")
    print(f"Local Efficiency: {nx.local_efficiency(G)}")
    print(f"Global Efficiency: {nx.global_efficiency(G)}")


def degree_histogram(G: nx.Graph, clamp_value=None):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    # print(degree_sequence)
    print("plotting...")
    # need to clamp values here
    if clamp_value:
        for i, x in enumerate(degree_sequence):
            if x > clamp_value:
                degree_sequence[i] = clamp_value

    plt.bar(*np.unique(degree_sequence, return_counts=True))
    locs, labels = plt.xticks()
    locs = locs[1:-1]  # check this?
    labels = [str(x) for x in locs]
    if clamp_value:
        labels[-1] = "> " + labels[-1]
    plt.xticks(locs, labels)
    print(locs)
    print(labels)
    # plt.ylim(0, 100)
    plt.show()


if __name__ == "__main__":
    # dataset = AutonomousSystems()
    # print_graph_metrics(dataset.G)
    dataset2 = Connectome()
    dataset2.generate_splits()
    # print_graph_metrics(dataset2.G)
    # dataset3 = Facebook()
    #  print_graph_metrics(dataset3.G)
    # degree_histogram(dataset3.G, clamp_value=100)
    # dataset = Connectome()
    # dataset.generate_splits()
    # G, sampled = dataset.splits[1]
    # write_edge_test_split(G, sampled, train_name="c_train.txt", test_name="c_test.txt")
