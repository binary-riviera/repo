# core training script
from preprocess import load_graph
from models.classical import adamic_adar, jaccard_coefficient, preferential_attachment
from models.kronecker import kronecker
from models.stochastic_block import stochastic_block_model
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

# sys.path.append("./libraries/pysbm/pybsm")


def generate_observed_graph(G: nx.Graph, n_samples: int):
    # TODO: allow for percentage to be passed instead?
    # print(f"Sampling {n_samples} edges from graph...")
    sampled_edges = random.sample(G.edges, n_samples)
    G_observed = G.copy()
    G_observed.remove_edges_from(sampled_edges)
    # print(G_observed.number_of_nodes())
    return (G_observed, sampled_edges)


def generate_observed_graph_connected(
    G: nx.Graph, n_samples: int, max_component_limit=3
):
    new_G = G.copy()
    sampled_edges = []
    while n_samples > 0:
        edge_to_remove = random.choice(list(new_G.edges))
        new_G.remove_edge(*edge_to_remove)
        if nx.number_connected_components(new_G) <= max_component_limit:
            sampled_edges.append(edge_to_remove)
            n_samples -= 1
        else:
            new_G.add_edge(*edge_to_remove)
            continue
    return (new_G, sampled_edges)


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
            try:
                if model == "adamic":
                    fpr, tpr = generate_roc_curve(adamic_adar(G), sampled)
                elif model == "jaccard":
                    fpr, tpr = generate_roc_curve(jaccard_coefficient(G), sampled)
                elif model == "preferential":
                    fpr, tpr = generate_roc_curve(preferential_attachment(G), sampled)
                elif model == "kronecker":
                    fpr, tpr = generate_roc_curve(kronecker(G), sampled)
                elif model == "sbm_standard":
                    fpr, tpr = generate_roc_curve(
                        stochastic_block_model(G, model_type="standard"), sampled
                    )
                elif model == "sbm_degree_corrected":
                    fpr, tpr = generate_roc_curve(
                        stochastic_block_model(G, model_type="degree_corrected"),
                        sampled,
                    )
                acc_results.append([fpr, tpr, model])
                console.log(f"finished running model {model}")
            except Exception:
                console.print_exception()

    plot_rocs(*acc_results)


def main():
    graph_name = "connectome"
    print(f"Loading graph {graph_name}...")
    G = load_graph(graph_name)  # TODO: make sure graph is undirected?

    print(f"Loaded graph with nodes: {G.number_of_nodes()} edges {G.number_of_edges()}")
    # foo = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    percent_to_remove = 0.05  # was 0.05
    print(G.number_of_edges() * percent_to_remove)
    G_observed, sampled = generate_observed_graph_connected(
        G,
        int(G.number_of_edges() * percent_to_remove),
        max_component_limit=1,
    )
    print(
        f"Observation graph has nodes: {G_observed.number_of_nodes()}, edges: {G_observed.number_of_edges()}"
    )
    print(f"{nx.number_connected_components(G_observed)} connected components")

    run_models(
        [
            "sbm_standard",
            "adamic",
            "jaccard",
            "preferential",
            "sbm_degree_corrected",
            "kronecker",
        ],
        G_observed,
        sampled,
    )


if __name__ == "__main__":
    main()
