# attempting to just wrap the connectome data and see whether SAGE will even work

# 0: Imports
import pandas as pd
import dgl
from dgl.data import DGLDataset
import torch
import os
import numpy as np
import dgl.function as fn
import scipy.sparse as sp
import torch.nn.functional as F
from dgl.nn import SAGEConv
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
import itertools

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 1: create dataset


class ConnectomeDataset(
    DGLDataset
):  # TODO: make this use the same test/train split as the other methods
    def __init__(self):
        super().__init__(name="connectome")

    def process(self):
        df = pd.read_csv("data/Connectome_matrix.csv", index_col=0)
        # need to turn it into format
        # | Src | Dst | Weight
        # NOTE: I'm assuming the IDs of the nodes don't matter here

        np_mtx = df.to_numpy(dtype=float)

        edges_data = pd.DataFrame(
            [
                {"Src": u, "Dst": v, "Weight": weight}
                for ((u, v), weight) in np.ndenumerate(np_mtx)
                if weight > 0
            ]
        )
        print(edges_data)

        # FIXME: the number of edges here don't match the number networkx gets?
        # edge_features = torch.from_numpy(edges_data["Weight"].to_numpy())
        edges_src = torch.from_numpy(edges_data["Src"].to_numpy())
        edges_dst = torch.from_numpy(edges_data["Dst"].to_numpy())

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=df.shape[0])
        # self.graph.edata["weight"] = edge_features

        self.graph.ndata["feat"] = torch.from_numpy(np_mtx.astype(np.float32))

        # print(self.graph.in_edges(1))

        # just doing some wacky stuff below
        # self.graph.update_all(
        #    lambda edge: {"m": edge.data["weight"]}, fn.sum("m", "h_f")
        # )

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    )
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    return roc_auc_score(labels, scores)


def compute_roc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    fpr, tpr, _ = roc_curve(labels, scores)
    return (fpr, tpr)


# 5: training


# what I need to do here is
# take a graph G


def train(test_percentage: float):
    # 1: prepare the training and testing sets
    dataset = ConnectomeDataset()
    g = dataset[0]
    g.remove_nodes(2727)  # 2728 is an isolated node, which seeems to screw things up
    u, v = g.edges()
    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.25)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
    train_g = dgl.remove_edges(g, eids[:test_size])

    # 2: create model
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    model = GraphSAGE(train_g.ndata["feat"].shape[1], 16)
    pred = DotPredictor()

    # 3: set up loss and optimizer
    optimizer2 = torch.optim.Adam(
        itertools.chain(model.parameters(), pred.parameters()), lr=0.01
    )
    optimizer = torch.optim.AdamW(
        itertools.chain(model.parameters(), pred.parameters()), lr=0.01
    )
    # 4: training
    all_logits = []
    for e in range(150):
        # forward
        h = model(train_g, train_g.ndata["feat"])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print("In epoch {}, loss: {}".format(e, loss))

    # 5: return ROC curve
    with torch.no_grad():
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        print("AUC", compute_auc(pos_score, neg_score))
        return compute_roc(pos_score, neg_score)
