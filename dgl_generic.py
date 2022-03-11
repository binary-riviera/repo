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
import networkx as nx


class Dataset(DGLDataset):
    def __init__(self, original_G, sampled):
        self.original_G = original_G
        self.sampled = sampled
        super().__init__("dataset")

    def process(self):
        ul = []
        vl = []
        self.indexes_of_sampled = []
        self.inv = []
        # v. crude approach but easiest to implement
        i = 0
        for (u, v) in self.original_G.edges():
            ul.append(u)
            vl.append(v)
            if (u, v) in self.sampled:
                self.indexes_of_sampled.append(i)
            else:
                self.inv.append(i)
            i += 1

        edges_src = torch.from_numpy(np.array(ul))
        edges_dst = torch.from_numpy(np.array(vl))
        np_mtx = nx.to_numpy_array(self.original_G)
        # print(np_mtx.shape)
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=np_mtx.shape[0])
        self.graph.ndata["feat"] = torch.from_numpy(np_mtx.astype(np.float32))

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


def construct_negative_graph(graph, k):
    src, dst = graph.edges()

    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,))
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())


def train(ds: Dataset, sampled):
    g = ds[0]

    # need to generate test_pos_g, then should be able to use construct_negative_graph to get test_neg_g
    u, v = g.edges()

    test_pos_u = [u[i] for i in ds.indexes_of_sampled]
    test_pos_v = [v[i] for i in ds.indexes_of_sampled]
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())

    train_pos_u = [u[i] for i in ds.inv]
    train_pos_v = [v[i] for i in ds.inv]
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())

    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))

    print(f"{len(np.ones(len(u)))=}")
    print(len(u.numpy()))
    print(len(v.numpy()))
    print(adj.todense().shape)

    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    test_size = len(ds.indexes_of_sampled)
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    train_g = dgl.remove_edges(g, ds.indexes_of_sampled)

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
    for e in range(500):
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
        return compute_auc(pos_score, neg_score)


def run_sage(original_G: nx.Graph, sampled):
    ds = Dataset(original_G, sampled)
    return train(ds, sampled)


if __name__ == "__main__":
    import datasets

    ds = datasets.Facebook()
    ds.G = nx.convert_node_labels_to_integers(ds.G)
    ds.G = ds.G.to_directed()
    # print(nx.number_connected_components(ds.G))
    original_G = ds.G
    # print(original_G)
    ds.generate_splits(end_p=0.15)
    _, sampled = ds.splits[0]
    # print(original_G)
    # print(len(sampled))
    wow = Dataset(original_G, sampled)
    train(wow, sampled)
