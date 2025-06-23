import argparse
import numpy as np
import scipy.sparse as sp
from scipy.linalg import fractional_matrix_power, inv, expm
import torch
import random
import networkx as nx
import dgl
from dgl import DGLGraph
from dgl.data import *


def load_npz_to_sparse_graph(file_name):

    with np.load('dataset/' + file_name + '.npz', allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']), shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']), shape=loader['labels_shape'])
        elif 'labels' in loader:
            labels = loader['labels']
        else:
            labels = None

        adj = adj_matrix.todense()
        col, row = np.where(adj>0)

        feat = attr_matrix.todense()
        num_class = len(set(labels))

        graph = dgl.graph((col, row), num_nodes=adj.shape[0])
        graph.ndata['feat'] = torch.FloatTensor(feat)
        graph.ndata['label'] = torch.LongTensor(labels)

    return graph, num_class


def get_split(g, nclass, train=20, valid=30):

    label = g.ndata['label'].numpy().tolist()

    class_ind = [[] for i in range(nclass)]
    for ind, lab in enumerate(label):
        class_ind[lab].append(ind)

    train_ind = []
    val_ind = []
    test_ind = []

    for i in range(nclass):
        inds = class_ind[i]
        random.shuffle(inds)
        train_ind.extend(inds[:train])
        val_ind.extend(inds[train:train+valid])
        test_ind.extend(inds[train+valid:])

    train_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
    val_mask   = torch.zeros(g.num_nodes(), dtype=torch.bool)
    test_mask  = torch.zeros(g.num_nodes(), dtype=torch.bool)

    train_mask[torch.LongTensor(train_ind)] = 1
    val_mask[torch.LongTensor(val_ind)]     = 1
    test_mask[torch.LongTensor(test_ind)]   = 1

    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask']   = val_mask
    g.ndata['test_mask']  = test_mask

    return g


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(name, split='random', seed=0, **kwargs):
    random.seed(seed)

    nclass=None
    graph=None
    if name in ['cora', 'citeseer', 'pubmed',"photo","computers"]:
        if name == 'cora':
            dataset = CoraGraphDataset(verbose=False)
        if name == 'citeseer':
            dataset = CiteseerGraphDataset(verbose=False)
        if name == 'pubmed':
            dataset = PubmedGraphDataset(verbose=False)
        if name=="photo":
            dataset = AmazonCoBuyPhotoDataset(verbose=False)
        if name == "computers":
            dataset =AmazonCoBuyComputerDataset(verbose=False)
        nclass = dataset.num_classes
        graph = dataset[0]

        graph = dgl.to_bidirected(graph, copy_ndata=True)
        graph = dgl.add_self_loop(graph)
        graph = dgl.to_simple(graph, copy_ndata=True)
        adj = graph.adj()
        #adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = graph.ndata['feat']
        #features = normalize_features(features)
        labels = graph.ndata['label']

    return adj, features, labels
    # return graph, nclass

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def random_planetoid_splits(y, percls_trn=20, val_lb=500, Flag=1):
    num_classes = len(set(y.tolist()))
    num_nodes = len(y)
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    #percls_trn = int(round(0.6 * num_nodes / num_classes))
    #val_lb = int(round(0.2 * num_nodes))

    indices = []
    for i in range(num_classes):
        index = (y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag is 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        # rest_index = rest_index[torch.randperm(rest_index.size(0))]

        train_mask = index_to_mask(train_index, size=num_nodes)
        val_mask = index_to_mask(rest_index[:val_lb], size=num_nodes)
        test_mask = index_to_mask(rest_index[val_lb:], size=num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        train_mask = index_to_mask(train_index, size=num_nodes)
        val_mask = index_to_mask(val_index, size=num_nodes)
        test_mask = index_to_mask(rest_index, size=num_nodes)
    return train_mask,val_mask,test_mask

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1,dtype=float).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
