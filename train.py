# core training script
from json import load
from preprocess import load_graph
from models.classical import adamic_adar, jaccard_coefficient, preferential_attachment
from models.kronecker import kronecker
from models.spectral_clustering import run_spectral_clustering
from sklearn.metrics import roc_curve, roc_auc_score
from rich.console import Console
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np


console = Console()

# config options
# SEED = 123456
# random.seed(a=SEED, version=2)


def generate_observed_graph(G: nx.Graph, n_samples: int):
    # TODO: allow for percentage to be passed instead?
    # print(f"Sampling {n_samples} edges from graph...")
    sampled_edges = random.sample(G.edges, n_samples)
    G_observed = G.copy()
    G_observed.remove_edges_from(sampled_edges)
    # print(G_observed.number_of_nodes())
    return (G_observed, sampled_edges)


def generate_roc_curve(edge_probs, removed_edges):
    # edge_probs are of the form (u, v, prob)
    # roc_curve expects a label (True or False depending on whether the edge actually exists)
    # and a score: the prob
    score, label = zip(*[(s, (u, v) in removed_edges) for (u, v, s) in edge_probs])
    results = roc_curve(label, score)
    return (results[0], results[1])  # we don't particularly care about the thresholds


def plot_rocs(*datasets, plot_baseline=True):
    print(f"plotting {len(datasets)} results...")
    if plot_baseline:
        plt.plot(np.arange(0.0, 1.0, 0.01), np.arange(0.0, 1.0, 0.01), label="baseline")
    for dataset in datasets:
        # print(dataset)
        plt.plot(dataset[0], dataset[1], label=dataset[2])
    plt.legend(loc="best")
    plt.show()


def run_models(models, G, sampled):
    acc_results = []
    for model in models:
        with console.status(f"[bold green]Training model {model}...") as status:
            # at the moment we'll just get the accuracy score
            fpr = []
            tpr = []
            if model == "adamic":
                fpr, tpr = generate_roc_curve(adamic_adar(G), sampled)
            elif model == "jaccard":
                fpr, tpr = generate_roc_curve(jaccard_coefficient(G), sampled)
            elif model == "preferential":
                fpr, tpr = generate_roc_curve(preferential_attachment(G), sampled)
            elif model == "kronecker":
                fpr, tpr = generate_roc_curve(kronecker(G), sampled)
            elif model == "spectral_clustering":
                run_spectral_clustering(G)

            acc_results.append([fpr, tpr, model])
            console.log(f"finished running model {model}")

    plot_rocs(*acc_results)


def main():
    graph_name = "connectome"
    print(f"Loading graph {graph_name}...")
    G = load_graph(graph_name)  # TODO: make sure graph is undirected?
    print(f"Loaded graph with nodes: {G.number_of_nodes()} edges {G.number_of_edges()}")
    percent_to_remove = 0.05
    print(G.number_of_edges() * percent_to_remove)
    G_observed, sampled = generate_observed_graph(
        G, int(G.number_of_edges() * percent_to_remove)
    )
    print(
        f"Observation graph has nodes: {G_observed.number_of_nodes()}, edges: {G_observed.number_of_edges()}"
    )
    run_models(
        ["spectral_clustering", "adamic", "jaccard", "preferential", "kronecker"],
        G_observed,
        sampled,
    )


if __name__ == "__main__":
    main()
