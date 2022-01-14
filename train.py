# core training script
from json import load
from preprocess import load_graph
from models.classical import adamic_adar
from sklearn.metrics import roc_curve, roc_auc_score
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np


def generate_observed_graph(G: nx.Graph, n_samples: int):
    # TODO: allow for percentage to be passed instead?
    sampled_edges = random.sample(G.edges, n_samples)
    G_observed = G.copy()
    G_observed.remove_edges_from(sampled_edges)
    return (G_observed, sampled_edges)


def generate_roc_curve(edge_probs, removed_edges):
    # edge_probs are of the form (u, v, prob)
    # roc_curve expects a label (True or False depending on whether the edge actually exists)
    # and a score: the prob
    score, label = zip(*[(s, (u, v) in removed_edges) for (u, v, s) in edge_probs])
    return roc_curve(label, score)


def plot_rocs(*datasets, plot_baseline=True):
    if plot_baseline:
        plt.plot(np.arange(0.0, 1.0, 0.01), np.arange(0.0, 1.0, 0.01), label="baseline")
    for dataset in datasets:
        plt.plot(dataset[0], dataset[1], label=dataset[2])
    plt.legend(loc="best")
    plt.show()


def main():
    graph_name = "karate"
    print(f"Loading graph {graph_name}...")
    G = load_graph(graph_name)  # TODO: make sure graph is undirected?
    percent_to_remove = 0.25
    G_observed, sampled = generate_observed_graph(
        G, int(G.number_of_edges() * percent_to_remove)
    )
    predictions = adamic_adar(G_observed)
    fpr, tpr, _ = generate_roc_curve(edge_probs=predictions, removed_edges=sampled)
    plot_rocs([fpr, tpr, "adamic"])


if __name__ == "__main__":
    main()
