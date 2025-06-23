from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import csv
import random
from scipy import sparse
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.utils import load_data, accuracy
from src.model_confidence import NNet
from sklearn.preprocessing import StandardScaler

# 去均值和方差归一化
# 可以算均值，标准差， 标准差标准化矩阵
scaler = StandardScaler()
# Training settings
parser = argparse.ArgumentParser()
'''
name or flags - 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo。
action - 当参数在命令行中出现时使用的动作基本类型。action=store_true用于指定当命令行参数存在时，将其值设置为True。如果命令行参数不存在，则该值将保持为默认值（通常为False）。
nargs - 命令行参数应当消耗的数目。
const - 被一些 action 和 nargs 选择所需求的常数。
default - 当参数未在命令行中出现时使用的值。   
type - 命令行参数应当被转换成的类型。
choices - 可用的参数的容器。
required - 此命令行选项是否可省略 （仅选项可用）。
help - 一个此选项作用的简单描述。
metavar - 在使用方法消息中使用的参数值示例。
dest - 被添加到 parse_args() 所返回对象上的属性名。
'''
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=50, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.2,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--input_droprate', type=float, default=0.6,
                    help='Dropout rate of the input layer (1 - keep probability).')
parser.add_argument('--hidden_droprate', type=float, default=0.8,
                    help='Dropout rate of the hidden layer (1 - keep probability).')

parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--K', type=int, default=7, help='Propagation step')
parser.add_argument('--sample', type=int, default=3, help='Sampling times of dropnode')
parser.add_argument('--tem', type=float, default=0.2, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1.0, help='Lamda')
parser.add_argument('--dataset', type=str, default='pubmed', help='Data set')
parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device')
# Batch Normalization的目的就是使我们的feature map满足均值为0，方差为1的分布规律。 均值方差都是向量，是每一个维度的均值方差，然后再进行标准化
parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')
# dataset = 'citeseer'
# dataset = 'pubmed'
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.set_device(0)
dataset = args.dataset

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
A, features, labels, idx_train, idx_val, idx_test = load_data(dataset)
idx_unlabel = torch.range(idx_train.shape[0], labels.shape[0] - 1, dtype=int)

feature = torch.FloatTensor(features)
non_zero_counts = torch.count_nonzero(feature, dim=1)
non_zero_counts = non_zero_counts / non_zero_counts.numel()
non_zero_counts = non_zero_counts * 5
non_zero_counts = torch.unsqueeze(non_zero_counts, dim=1)
f = torch.mul(feature, non_zero_counts)
f = f.cuda()


A = A.cuda()
# Model and optimizer
model = NNet(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            input_droprate=args.input_droprate,
            hidden_droprate=args.hidden_droprate,
            use_bn=args.use_bn,
            K = args.K)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    A = A.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    idx_unlabel = idx_unlabel.cuda()

def consis_loss(logps, temp=args.tem):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p / len(ps)
    # p2 = torch.exp(logp2)

    sharp_p = (torch.pow(avg_p, 1. / temp) / torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p - sharp_p).pow(2).sum(1))
    loss = loss / len(ps)
    return args.lam * loss

def consis(logps, temp=args.tem):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p / len(ps)
    return avg_p


def kl_div(predict, soft_target):
    return 0.01*F.kl_div(soft_target+1e-15, predict, reduction="batchmean")

def train(epoch):
    t = time.time()
    X = features
    X_agu = f
    model.train()
    optimizer.zero_grad()
    K = args.sample
    output_list = []
    for k in range(K):
        output_list.append(torch.log_softmax(model(X+X_agu, A), dim=-1))

    loss_train = 0.
    for k in range(K):
        loss_train += F.nll_loss(output_list[k][idx_train], labels[idx_train])

    loss_train = loss_train / K
    output2 = consis(output_list)
    output3 = model.mlppp(X)
    loss_kl = kl_div(output2, output3)
    loss_consis = consis_loss(output_list)
    loss_train = loss_train + loss_consis + loss_kl
    acc_train = accuracy(output_list[0][idx_train], labels[idx_train])
    # if output_list[0].grad_fn is None:
    #     raise RuntimeError("张量没有梯度函数")
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(X, A)
        output = torch.log_softmax(output, dim=-1)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.item(), acc_val.item()


def Train():
    # Train model
    t_total = time.time()
    loss_values = []
    acc_values = []
    bad_counter = 0
    # best = args.epochs + 1
    loss_best = np.inf
    acc_best = 0.0

    loss_mn = np.inf
    acc_mx = 0.0
    sum_ep = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        # if epoch < 200:
        #   l, a = train(epoch, True)
        #   loss_values.append(l)
        #   acc_values.append(a)
        #   continue
        sum_ep += 1
        l, a = train(epoch)
        loss_values.append(l)
        acc_values.append(a)

        print(bad_counter)

        if loss_values[-1] <= loss_mn or acc_values[-1] >= acc_mx:  # or epoch < 400:
            if loss_values[-1] <= loss_best:  # and acc_values[-1] >= acc_best:
                loss_best = loss_values[-1]
                acc_best = acc_values[-1]
                best_epoch = epoch
                torch.save(model.state_dict(), dataset + '.pkl')

            loss_mn = np.min((loss_values[-1], loss_mn))
            acc_mx = np.max((acc_values[-1], acc_mx))
            bad_counter = 0
        else:
            bad_counter += 1

        # print(bad_counter, loss_mn, acc_mx, loss_best, acc_best, best_epoch)
        if bad_counter == args.patience:
            print('Early stop! Min loss: ', loss_mn, ', Max accuracy: ', acc_mx)
            print('Early stop model validation loss: ', loss_best, ', accuracy: ', acc_best)
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load(dataset + '.pkl'))
    return a, best_epoch, sum_ep


def test():
    model.eval()
    X = features
    output = model(X, A)
    output = torch.log_softmax(output, dim=-1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test


def setseed(seed):
    torch.cuda.set_device(0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    pass


def muilt_run():
    filename = "venv/actor_newdata.csv"
    with open(filename, "a", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "value", "val_acc", "best_epoch", "last_epoch"])
        writer.writeheader()
        pass
    acc_list = []
    start_time = time.time()
    for i in range(16, 26):
        with open(filename, "a", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["id", "value", "val_acc", "best_epoch", "last_epoch"])
            # args.seed=20
            setseed(i)
            val_acc, best_epoch, last_epoch = Train()
            acc = test()
            acc_list.append(acc)
            print("*" * 20)
            print("训练次数为 i={:.4f}".format(i))
            writer.writerow(
                {"id": i, "value": acc, "val_acc": val_acc, "best_epoch": best_epoch, "last_epoch": last_epoch})

    mean = torch.round(torch.mean(torch.tensor(acc_list)), decimals=4)
    std = torch.round(torch.std(torch.tensor(acc_list)), decimals=4)
    print('mean', mean)
    print('std', std)
    end_time = time.time()
    with open(filename, "a", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["mean", "std"])
        writer.writeheader()
        writer.writerow({"mean": mean, "std": std})
        pass
    pass

muilt_run()
#Train()
#test()
