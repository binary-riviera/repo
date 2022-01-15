import subprocess
import networkx as nx

# TODO: move this to a config file
snap_dir = "./snap/snap/examples/kronem/"
command_name = "kronem"
kron_iters = 1


def extract_prob_matrix(loc):
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
        row1, row2 = init_str.split(";")
        row1 = row1.split(",")
        row2 = row2.split(",")
        matrix = [[float(row1[0]), float(row1[1])], [float(row2[0]), float(row2[1])]]
        return matrix


def calc_prob(u: int, v: int, mtx):
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
    dim = 2
    p = 1.0
    for _ in range(kron_iters):
        p *= mtx[u % dim][v % dim]
        if p == 0.0:
            return 0.0
        u //= dim
        v //= dim
    # print(f"prob {u}->{v} : {p}")
    return p


def get_probs(G: nx.graph, mtx):
    # needs to return (u, v, prob) like all other models
    probs = []
    edges = G.number_of_edges()
    for i in range(edges):
        for j in range(edges):
            if i != j and not G.has_edge(i, j):
                # we don't care about the probs for observed edges
                probs.append((i, j, calc_prob(i, j, mtx)))
    return probs


def train(stdout_file=None):
    command = snap_dir + command_name
    if stdout_file:
        with open(stdout_file, "w") as outfile:
            subprocess.call(
                [
                    command,
                    "-i:./data/as20graph.txt",
                    "-n0:2",
                    '-m:"0.9 0.6; 0.6 0.1"',
                    "-ei:50",
                ],
                stdout=outfile,
            )
    else:
        subprocess.call(
            [
                command,
                "-i:./data/as20graph.txt",
                "-n0:2",
                '-m:"0.9 0.6; 0.6 0.1"',
                "-ei:50",
            ],
        )


def kronecker(G: nx.graph):
    train(stdout_file="kronecker.log")
    mtx = extract_prob_matrix("./KronEM-as20graph.tab")
    return get_probs(G, mtx)


if __name__ == "__main__":
    kronecker()
