#  see pybsm
import pysbm
import networkx as nx
from dataclasses import dataclass
import pandas as pd
import numpy as np
from rich.console import Console


@dataclass
class StochasticBlockModel:
    """Stochastic Block Model"""

    G: nx.graph
    num_blocks: int = 2
    model_type: str = "standard"
    is_directed: bool = False
    num_iters: int = 100000  # default is 1000 I think

    def generate_stochastic_block(self):  # what if is_directed = True?
        standard_partition = pysbm.NxPartition(
            graph=self.G, number_of_blocks=self.num_blocks
        )
        if self.model_type == "standard":
            standard_objective_function = pysbm.TraditionalUnnormalizedLogLikelyhood(
                is_directed=self.is_directed
            )
            standard_inference = pysbm.MetropolisHastingInference(
                self.G, standard_objective_function, standard_partition
            )
            standard_inference.infer_stochastic_block_model(self.num_iters)
            self.block_edges = standard_partition.block_edges
            self.block_memberships = standard_partition.get_block_memberships()
        elif self.model_type == "degree_corrected":
            degree_corrected_partition = pysbm.NxPartition(
                graph=self.G,
                number_of_blocks=self.num_blocks,
                representation=standard_partition.get_representation(),
            )
            degree_corrected_objective_function = (
                pysbm.DegreeCorrectedUnnormalizedLogLikelyhood(
                    is_directed=self.is_directed
                )
            )

            degree_corrected_inference = pysbm.MetropolisHastingInference(
                self.G,
                degree_corrected_objective_function,
                degree_corrected_partition,
            )
            degree_corrected_inference.infer_stochastic_block_model(self.num_iters)
            self.block_edges = degree_corrected_partition.block_edges
            self.block_memberships = degree_corrected_partition.get_block_memberships()
        elif self.model_type == "hierarchical":
            # probably isn't implemented fully correctly
            hierarchical_partition = pysbm.NxHierarchicalPartition(
                self.G, number_of_blocks=self.num_blocks
            )
            hierarchical_objective_function = (
                pysbm.LogLikelihoodOfHierarchicalMicrocanonicalNonDegreeCorrected(
                    is_directed=self.is_directed
                )
            )
            hierarchical_inference = pysbm.PeixotoHierarchicalInference(
                self.G, hierarchical_objective_function, hierarchical_partition
            )
            hierarchical_inference.infer_stochastic_block_model(self.num_iters)
            self.block_edges = hierarchical_partition.block_edges
            self.block_memberships = hierarchical_partition.get_block_memberships()

    def scale_edges_to_probs(self):
        # this whole function is probably not the correct approach
        # but it should approximate the likelihood well enough for now
        mat = self.block_edges
        self.prob_mtx = (mat - np.min(mat)) / (np.max(mat) - np.min(mat))

    def get_probs(self):
        # needs to return (u, v, prob) like all other models
        probs = []
        nodes = self.G.number_of_nodes()
        for i in range(nodes):
            for j in range(nodes):
                if i != j and not self.G.has_edge(i, j):
                    # we don't care about the probs for observed edges
                    block_i = self.block_memberships[i]
                    block_j = self.block_memberships[j]
                    p = self.prob_mtx[block_i][block_j]
                    probs.append((i, j, p))
        return probs

    def run(self):
        self.generate_stochastic_block()
        self.scale_edges_to_probs()
        return self.get_probs()


def stochastic_block_model(G: nx.graph, num_blocks=50, model_type="standard"):
    # foo = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    # print(foo)

    sbm = StochasticBlockModel(G, num_blocks=num_blocks, model_type=model_type)
    probs = sbm.run()
    return probs


if __name__ == "__main__":
    df = pd.read_csv("../data/Connectome_matrix.csv", index_col=0)
    np_arr = df.to_numpy(dtype=np.integer)
    G = nx.convert_matrix.from_numpy_array(np_arr)
    stochastic_block_model(G)
    # stochastic_block_model(nx.karate_club_graph())
