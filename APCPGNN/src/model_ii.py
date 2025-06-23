import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init, Linear
import numpy as np
from src.layers import MLPLayer


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, is_cuda=True, use_bn=False):
        super(MLP, self).__init__()

        self.layer1 = MLPLayer(nfeat, nhid)
        self.layer2 = MLPLayer(nhid, nclass)

        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.is_cuda = is_cuda
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn

    def forward(self, x):

        if self.use_bn:
            x = self.bn1(x)
        x = F.dropout(x, self.input_droprate, training=self.training)
        x = F.relu(self.layer1(x))
        if self.use_bn:
            x = self.bn2(x)
        x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.layer2(x)

        return x


class Prop(nn.Module):
    def __init__(self, K, nclass):
        super().__init__()
        self.K = K
        self.s = Linear(nclass, 1)

    def forward(self, x, adj):
        preds = []
        preds.append(x)
        for i in range(self.K):
            x = torch.spmm(adj, x)
            preds.append(x)
        preds_stack = torch.stack(preds)

        sigmoid = torch.nn.Sigmoid()
        # relu = torch.nn.ReLU()
        preds_stack = sigmoid(preds_stack)
        # 计算包含大于0.5元素的行数
        counts = (preds_stack > 0.41).any(dim=2).sum(dim=1).float()
        # 对counts进行归一化
        counts_normalized = counts / counts.max()
        counts_normalized = counts_normalized.unsqueeze(1).unsqueeze(1)
        # 将归一化后的counts值与preds中的每个张量相乘
        normalized_preds = preds_stack * counts_normalized
        # 将normalized_preds中的张量对应相加
        result = normalized_preds.sum(dim=0)
        # result = relu(result)
        return result


class NNet(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, is_cuda=True, use_bn=False, K=0):
        super().__init__()
        self.layer1 = MLPLayer(nfeat, nhid)
        self.layer2 = MLPLayer(nhid, nclass)
        self.prop = Prop(K, nclass)

        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.is_cuda = is_cuda
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.K = K



    def forward(self, x, adj):

        if self.use_bn:
            x = self.bn1(x)
        x = F.dropout(x, self.input_droprate, training=self.training)
        x = F.relu(self.layer1(x))
        if self.use_bn:
            x = self.bn2(x)
        x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.layer2(x)

        x = self.prop(x, adj)
        return x
