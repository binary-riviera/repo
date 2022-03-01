# core training script
from preprocess import load_graph
from models.classical import adamic_adar, jaccard_coefficient, preferential_attachment
from models.kronecker import kronecker
from models.stochastic_block import stochastic_block_model
from sklearn.metrics import roc_curve, auc
from rich.console import Console
from rich.table import Table
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from dgl_connectome import train as train_dgl
from datasets import *


console = Console()

# config options
# SEED = 123456
# random.seed(a=SEED, version=2)

# sys.path.append("./libraries/pysbm/pybsm")


def debug(x):
    print(x)
    return x


def generate_observed_graph(G: nx.Graph, n_samples: int):
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


def maximise_auc_sbm(G, sampled):
    num_iters = [
        100,
        1000,
        2000,
        5000,
        10000,
        20000,
        50000,
        100000,
        200000,
        # 500000,
        # 1000000,
    ]
    num_blocks = [
        2,
        3,
        4,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
        100,
        150,
        200,
    ]
    max_auc = 0
    max_num_blocks = 0
    max_num_iters = 0
    for num_block in num_blocks:
        for num_iter in num_iters:
            fpr, tpr = generate_roc_curve(
                stochastic_block_model(
                    G, model_type="standard", num_blocks=num_block, num_iters=num_iter
                ),
                sampled,
            )
            a = auc(fpr, tpr)
            if a > max_auc:
                max_auc = a
                max_num_blocks = num_block
                max_num_iters = num_iter
            print(f"Checked blocks: {num_block} iters: {num_iter} auc:{a}")

    print(
        f"max auc was {max_auc}, with iters {max_num_iters}, blocks: {max_num_blocks}"
    )


def kronecker_auc_check(G, sampled):
    print("training")
    kronecker(G, kron_iters=1, train=True)  # generate the matrix
    print("calculating")
    for i in range(20):
        K = kronecker(G, kron_iters=i, train=False)
        fpr, tpr = generate_roc_curve(K, sampled)
        a = auc(fpr, tpr)
        print(f"kronIters: {i}  auc: {a}")


def plot_curves(*datasets, plot_baseline=True, ylims=None, xlims=None):
    print(f"plotting {len(datasets)} results...")
    if plot_baseline:
        plt.plot(np.arange(0.0, 1.0, 0.01), np.arange(0.0, 1.0, 0.01), label="baseline")
    for dataset in datasets:
        # print(dataset)
        plt.plot(dataset[0], dataset[1], label=dataset[2])
    plt.legend(loc="best")

    if ylims:
        plt.ylim([*ylims])
    if xlims:
        plt.xlim([*xlims])

    plt.show()


def run_models(models, G, sampled):
    acc_results = []
    table = Table(title="Model Results")
    table.add_column("Name", justify="left")
    table.add_column("AUC", justify="left", style="green")
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
                elif model == "sbm_hierarchical":
                    fpr, tpr = generate_roc_curve(
                        stochastic_block_model(G, model_type="hierarchical"), sampled
                    )
                elif model == "dgl":
                    fpr, tpr = train_dgl(0.25)
                acc_results.append([fpr, tpr, model])
                table.add_row(model, str(auc(fpr, tpr)))
                console.log(f"finished running model {model}")
            except Exception:
                console.print_exception()

    console.print(table)
    plot_curves(*acc_results)


def main():
    graph_name = "connectome"
    G = load_graph(graph_name)
    console.log(
        f"loaded graph '{graph_name}' with nodes: {G.number_of_nodes()} edges {G.number_of_edges()}"
    )

    percent_to_remove = 0.25  # was 0.05
    num_edges_to_remove = int(G.number_of_edges() * percent_to_remove)
    G_observed, sampled = generate_observed_graph_connected(
        G,
        num_edges_to_remove,
        max_component_limit=1,
    )
    console.log(
        f"generated observation graph with nodes: {G_observed.number_of_nodes()}, edges: {G_observed.number_of_edges()}. Removed {num_edges_to_remove} edges"
    )

    run_models(
        [
            "sbm_standard",
            "adamic",
            "jaccard",
            "preferential",
            "sbm_degree_corrected",
            "dgl",
            # "kronecker",
        ],
        G_observed,
        sampled,
    )


def main2():
    graph_name = "connectome"
    G = load_graph(graph_name)
    console.log(
        f"loaded graph '{graph_name}' with nodes: {G.number_of_nodes()} edges {G.number_of_edges()}"
    )
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    z = generate_graphs(G)

    ds = []
    print("foo")
    ds = [auc(*generate_roc_curve(adamic_adar(G), s)) for (G, s) in z]
    print("aa")
    ds_2 = [auc(*generate_roc_curve(jaccard_coefficient(G), s)) for (G, s) in z]
    print("jaccard")
    ds_3 = [auc(*generate_roc_curve(preferential_attachment(G), s)) for (G, s) in z]
    print("preferential")
    # ds.append([auc(*generate_roc_curve(kronecker(G), s)) for (G, s) in z])

    # ds.append(
    #    [
    #        auc(
    #            *generate_roc_curve(stochastic_block_model(G, model_type="standard"), s)
    #        )
    #        for (G, s) in z
    #    ]
    # )

    labels = ["aa", "jaccard"]

    p_r = list(np.arange(0.05, 0.45, 0.05))

    # n_ds = [(x, p_r, labels[idx]) for (idx, x) in enumerate(ds)]

    aa = (p_r, ds, "aa")
    jaccard = (p_r, ds_2, "jaccard")
    preferential = (p_r, ds_3, "preferential")
    plot_curves(aa, jaccard, preferential, plot_baseline=False, ylims=(0, 1.0))


def get_auc_curves(dataset: Dataset):
    # returns [([float],[float],str)]
    curves = []
    p_r = list(np.arange(0.05, 0.45, 0.05))

    ## Classical Approaches
    mod_splits = [(G.to_undirected(), s) for (G, s) in dataset.splits]
    # mod_splits = dataset.splits

    # Stochastic Block Model
    # print("generating curves for stochastic block model")
    # ds = [
    #    auc(*generate_roc_curve(stochastic_block_model(G, model_type="standard"), s))
    #    for (G, s) in mod_splits
    # ]
    # curves.append((p_r, ds, "sbm"))

    # Kronecker
    print("generating curves for kronecker")
    ds = [
        auc(*generate_roc_curve(kronecker(G), s)) for (G, s) in dataset.splits
    ]  # mod_splits]
    curves.append((p_r, ds, "kronecker"))

    # Adamic Adar
    print("generating curves for Adamic Adar")
    ds = [auc(*generate_roc_curve(adamic_adar(G), s)) for (G, s) in mod_splits]
    curves.append((p_r, ds, "aa"))

    # Jaccard
    print("generating curves for Jaccard")
    ds = [auc(*generate_roc_curve(jaccard_coefficient(G), s)) for (G, s) in mod_splits]
    curves.append((p_r, ds, "jaccard"))

    print("generated curves")

    return curves


def main3():

    print("loading dataset")
    ds = Connectome()

    print("generating splits")
    # ds.G = ds.G.to_undirected()
    ds.generate_splits(keep_connected=True)

    print("Generating Curves")
    c = get_auc_curves(ds)

    print("Plotting graph...")

    plot_curves(*c, plot_baseline=False, ylims=(0, 1.0))


if __name__ == "__main__":
    main3()
