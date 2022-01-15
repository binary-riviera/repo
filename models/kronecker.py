import subprocess
import networkx as nx
import os.path

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
    nodes = G.number_of_nodes()
    for i in range(nodes):
        for j in range(nodes):
            if i != j and not G.has_edge(i, j):
                # we don't care about the probs for observed edges
                probs.append((i, j, calc_prob(i, j, mtx)))
    return probs


def train(snap_filepath: str, stdout_file=None):
    command = snap_dir + command_name
    if stdout_file:
        with open(stdout_file, "w") as outfile:
            subprocess.call(
                [
                    command,
                    "-i:" + snap_filepath,
                    "-n0:2",
                    '-m:"0.9 0.6; 0.6 0.1"',
                    "-ei:150",
                ],
                stdout=outfile,
            )
    else:
        subprocess.call(
            [
                command,
                "-i:" + snap_filepath,
                "-n0:2",
                '-m:"0.9 0.6; 0.6 0.1"',
                "-ei:150",
            ],
        )


def write_graph_to_snap_format(G: nx.graph, data_dir: str, name: str):
    if os.path.isfile(data_dir + name):
        # file exists
        return
    # snap expects a graph of the format:
    # SrcNId DstNId
    with open(data_dir + name, "w") as snap_file:
        snap_file.write("# " + name + "\n")
        snap_file.write(
            f"# Nodes: {G.number_of_nodes()} Edges: {G.number_of_edges()}\n"
        )
        snap_file.write("# SrcNId\tDstNId\n")
        nodes = G.number_of_nodes()
        for i in range(nodes):
            for j in range(nodes):
                if G.has_edge(i, j):
                    # write the edge to file
                    snap_file.write(f"{i}\t{j}\n")


def kronecker(G: nx.graph, cleanup=False):
    log_file = "kronecker.log"
    data_dir = "./data/"
    snap_file = "connectome.txt"
    write_graph_to_snap_format(G, data_dir, snap_file)
    train(snap_filepath=data_dir + snap_file, stdout_file=log_file)
    mtx = extract_prob_matrix("./KronEM-as20graph.tab")

    if cleanup:
        pass  # TODO: implement

    return get_probs(G, mtx)


if __name__ == "__main__":
    kronecker()
