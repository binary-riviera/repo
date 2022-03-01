import subprocess
import networkx as nx
import os.path
from dataclasses import dataclass


@dataclass
class Kronecker:
    """Kronecker Expectation Maximisation"""

    G: nx.graph
    data_filepath: str
    snap_dir: str = "./snap/snap/examples/kronem/"
    command_name: str = "kronem"
    kron_iters: int = (
        13  # TODO: work out how snap calculates the kron_iters then replicate that here
    )
    matrix: list = None
    stdout_file: str = None
    # KRONECKER PARAMETERS
    initiator_matrix: str = (
        "R"  # "0.9 0.7; 0.5 0.2"  # Init Gradient Descent Matrix ('R' for random)
    )
    m_step_grad_iters: int = 10  # Gradient descent iterations for M-step
    em_iters: int = 150  # EM iterations
    min_grad_step: float = 0.001  # Minimum gradient step for M-step
    max_grad_step: float = 0.008  # Maximum graidient step for M-step
    warmup_samples: int = 5000  # Samples for MCMC warm-up
    grad_est_samples: int = 1500  # Samples per gradient estimation
    sim: bool = False  # Scale the initiator to match the number of edges
    nsp: float = 0.6  # Probability of using NodeSwap vs EdgeSwap MCMC
    debug: bool = False  # Debug mode
    matrix_size: int = 2  # Init Matrix Size, needs to be reflected in initiator matrix

    def extract_prob_matrix(self, loc):
        init_str = None
        with open(loc) as matrix_file:
            # the reason we reverse is because we want the last initiator generated
            for line in reversed(matrix_file.readlines()):  # HACK: pretty inefficient
                if "Estimated initiator" in line:
                    init_str = line
                    break
        if not init_str:
            print("Couldn't find initiator matrix!")
        else:
            init_str = init_str.replace("Estimated initiator", "").strip()
            init_str = init_str[1:-1]  # remove the brackets
            rows = init_str.split(";")
            matrix = [[float(x) for x in row.split(",")] for row in rows]
            # print(f"{len(matrix)} - {len(matrix[0])}\n {matrix}")
            assert len(matrix) == self.matrix_size
            # print(f"extracted matrix of size {self.matrix_size}")
            self.matrix = matrix

    def calc_prob(self, u: int, v: int):
        """C++ taken from Kronecker.cpp in Snap
        double TKronMtx::GetEdgeProb(int NId1, int NId2, const int& NKronIters) const {
        double Prob = 1.0;
        for (int level = 0; level < NKronIters; level++) {
            Prob *= At(NId1 % MtxDim, NId2 % MtxDim);
            if (Prob == 0.0) { return 0.0; }
            NId1 /= MtxDim;  NId2 /= MtxDim;
        }
        return Prob;
        }
        """
        dim = self.matrix_size  # 2
        p = 1.0
        for _ in range(self.kron_iters):
            p *= self.matrix[u % dim][v % dim]
            if p == 0.0:
                return 0.0
            u //= dim
            v //= dim
        # print(f"prob {u}->{v} : {p}")
        return p

    def get_probs(self):
        # needs to return (u, v, prob) like all other models
        probs = []
        nodes = self.G.number_of_nodes()
        for i in range(nodes):
            for j in range(nodes):
                if i != j and not self.G.has_edge(i, j):
                    # we don't care about the probs for observed edges
                    probs.append((i, j, self.calc_prob(i, j)))
        return probs

    def train(self):
        command = self.snap_dir + self.command_name
        args = [
            command,
            "-i:" + self.data_filepath,
            "-n0:" + str(self.matrix_size),
            "-m:" + self.initiator_matrix,
            "-ei:" + str(self.em_iters),
            "-gi:" + str(self.m_step_grad_iters),
        ]
        # print(args)
        if self.sim:
            args.append("-sim")

        if self.stdout_file:
            with open(self.stdout_file, "w") as outfile:
                subprocess.call(args, stdout=outfile)
        else:
            subprocess.call(args)

    def write_graph_to_snap_format(self):
        # if os.path.isfile(self.data_filepath):
        # file exists
        #    return
        # snap expects a graph of the format:
        # SrcNId DstNId
        with open(self.data_filepath, "w") as snap_file:
            snap_file.write("# " + self.data_filepath + "\n")
            snap_file.write(
                f"# Nodes: {self.G.number_of_nodes()} Edges: {self.G.number_of_edges()}\n"
            )
            snap_file.write("# SrcNId\tDstNId\n")
            nodes = self.G.number_of_nodes()
            for i in range(nodes):
                for j in range(nodes):
                    if self.G.has_edge(i, j):
                        # write the edge to file
                        snap_file.write(f"{i}\t{j}\n")

    def run(self, train=True):
        self.write_graph_to_snap_format()
        if train:
            self.train()
        self.extract_prob_matrix("./KronEM-connectome.tab")
        return self.get_probs()


def kronecker(G: nx.graph, cleanup=False, kron_iters=13, train=True):
    print("running kronecker with graph ", G, " ...")
    kron = Kronecker(
        G,
        data_filepath="./data/connectome.txt",
        stdout_file="kronecker.log",
        kron_iters=kron_iters,
        sim=False,
        matrix_size=4,
    )
    return kron.run(train)


if __name__ == "__main__":
    kronecker()
