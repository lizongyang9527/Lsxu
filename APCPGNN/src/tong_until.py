import sys

import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp

import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_data(dataset_str='cora'):
    if dataset_str in ['cora', 'citeseer', 'pubmed']:
        # 原始处理逻辑保持不变
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        features = normalize(features)

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        features = normalize(features)
        adj = normalize(adj + sp.eye(adj.shape[0]))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)
        idx_test = test_idx_range.tolist()

    elif dataset_str in ['cs', 'phy']:
        # CS数据集处理
        data = np.load(f'data/{dataset_str}.npz')
        adj = sp.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']),
                            shape=data['adj_shape'])
        features = sp.csr_matrix((data['attr_data'], data['attr_indices'], data['attr_indptr']),
                                 shape=data['attr_shape'])
        labels = data['labels']

        features = normalize(features)
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        # CS数据划分
        num_classes = len(np.unique(labels))
        indices = []
        for i in range(num_classes):
            index = (labels == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        train_index = torch.cat([i[:20] for i in indices], dim=0)
        val_index = torch.cat([i[20:50] for i in indices], dim=0)
        test_index = torch.cat([i[50:] for i in indices], dim=0)

        idx_train = train_index
        idx_val = val_index
        idx_test = test_index

    elif dataset_str in ['photo','computers']:
        # 加载Amazon数据集
        data = np.load(f'data/{dataset_str}.npz')

        # 构建邻接矩阵和特征矩阵
        adj = sp.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']),
                            shape=data['adj_shape'])
        features = sp.csr_matrix((data['attr_data'], data['attr_indices'], data['attr_indptr']),
                                 shape=data['attr_shape'])
        labels = data['labels']

        # 标准化处理
        features = normalize(features)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize(adj + sp.eye(adj.shape[0]))

        # 转换为PyTorch Tensor
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        # 创建与random_coauthor_amazon_splits完全一致的划分逻辑
        num_classes = labels.max().item() + 1
        indices = []
        for i in range(num_classes):
            index = (labels == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        train_index = torch.cat([i[:20] for i in indices], dim=0)
        val_index = torch.cat([i[20:50] for i in indices], dim=0)
        test_index = torch.cat([i[50:] for i in indices], dim=0)

        # 转换为mask格式（与原始函数一致）
        idx_train = index_to_mask(train_index, size=len(labels))
        idx_val = index_to_mask(val_index, size=len(labels))
        idx_test = index_to_mask(test_index, size=len(labels))

    else:
        raise ValueError(f'Unknown dataset: {dataset_str}')

    return adj, features, labels, idx_train, idx_val, idx_test


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

    # 下面的关系不大

