import sys
import os.path
import numpy as np
import scipy.sparse as ssp
import math
import random

# add to the path
sys.path.append("%s/pytorch_DGCNN" % os.path.dirname(os.path.realpath(__file__)))
from main import *

sys.path.append("%s/SEAL/Python" % os.path.dirname(os.path.realpath(__file__)))
from util_functions import *


def load_test_train_idxs():
    # Do not use in actual pipeline
    train_file = "/Users/alcibiades/Documents/University/Dissertation/repo/SEAL/Python/data/c_train.txt"
    test_file = "/Users/alcibiades/Documents/University/Dissertation/repo/SEAL/Python/data/c_test.txt"
    train_idx = np.loadtxt(train_file, dtype=int)
    test_idx = np.loadtxt(test_file, dtype=int)
    return (train_idx, test_idx)


def seal(
    train_idx,
    test_idx,
    max_train_num=100000,
    all_unknown_as_negative=False,
    hop=1,
    max_nodes_per_hop=None,
    no_parallel=False,
    batch_size=50,
):
    # I think it expects a numpy 2 x n NDArray
    train_pos = (train_idx[:, 0], train_idx[:, 1])
    test_pos = (test_idx[:, 0], test_idx[:, 1])
    # build observed network from train_links
    max_idx = np.max(train_idx)
    max_idx = max(max_idx, np.max(test_idx))
    net = ssp.csc_matrix(
        (np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])),
        shape=(max_idx + 1, max_idx + 1),
    )
    net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges
    net[np.arange(max_idx + 1), np.arange(max_idx + 1)] = 0  # remove self-loops
    # use provided train/test positive links, sample negative from net
    train_pos, train_neg, test_pos, test_neg = sample_neg(
        net,
        train_pos=train_pos,
        test_pos=test_pos,
        max_train_num=max_train_num,
        all_unknown_as_negative=all_unknown_as_negative,
    )

    """Train and apply classifier"""
    A = net.copy()  # the observed network
    A[test_pos[0], test_pos[1]] = 0  # mask test links
    A[test_pos[1], test_pos[0]] = 0  # mask test links
    A.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x

    train_graphs, test_graphs, max_n_label = links2subgraphs(
        A,
        train_pos,
        train_neg,
        test_pos,
        test_neg,
        hop,
        max_nodes_per_hop,
        None,
        no_parallel,
    )
    print("# train: %d, # test: %d" % (len(train_graphs), len(test_graphs)))

    cmd_args.gm = "DGCNN"
    cmd_args.sortpooling_k = 0.6
    cmd_args.latent_dim = [32, 32, 32, 1]
    cmd_args.hidden = 128
    cmd_args.out_dim = 0
    cmd_args.dropout = True
    cmd_args.num_class = 2
    cmd_args.mode = "cpu"
    cmd_args.num_epochs = 50
    cmd_args.learning_rate = 1e-4
    cmd_args.printAUC = True
    cmd_args.feat_dim = max_n_label + 1
    cmd_args.attr_dim = 0

    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
        k_ = int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1
        cmd_args.sortpooling_k = max(10, num_nodes_list[k_])
        print("k used in SortPooling is: " + str(cmd_args.sortpooling_k))

    classifier = Classifier()
    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    random.shuffle(train_graphs)
    val_num = int(0.1 * len(train_graphs))
    val_graphs = train_graphs[:val_num]
    train_graphs = train_graphs[val_num:]

    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    best_epoch = None

    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)
        classifier.train()
        avg_loss = loop_dataset(
            train_graphs,
            classifier,
            train_idxes,
            optimizer=optimizer,
            bsize=batch_size,
        )
        if not cmd_args.printAUC:
            avg_loss[2] = 0.0
        print(
            "\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m"
            % (epoch, avg_loss[0], avg_loss[1], avg_loss[2])
        )

        classifier.eval()
        val_loss = loop_dataset(val_graphs, classifier, list(range(len(val_graphs))))
        if not cmd_args.printAUC:
            val_loss[2] = 0.0
        print(
            "\033[93maverage validation of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m"
            % (epoch, val_loss[0], val_loss[1], val_loss[2])
        )
        if best_loss is None:
            best_loss = val_loss
        if val_loss[0] <= best_loss[0]:
            best_loss = val_loss
            best_epoch = epoch
            test_loss = loop_dataset(
                test_graphs, classifier, list(range(len(test_graphs)))
            )
            if not cmd_args.printAUC:
                test_loss[2] = 0.0
            print(
                "\033[94maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m"
                % (epoch, test_loss[0], test_loss[1], test_loss[2])
            )

    print(
        "\033[95mFinal test performance: epoch %d: loss %.5f acc %.5f auc %.5f\033[0m"
        % (best_epoch, test_loss[0], test_loss[1], test_loss[2])
    )

    return test_loss[2]  # the auc


if __name__ == "__main__":
    train_idx, test_idx = load_test_train_idxs()
    seal(train_idx, test_idx)
