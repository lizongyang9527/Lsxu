
import torch


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(y, percls_trn=20, val_lb=500, Flag=0):
    num_classes=len(set(y.tolist()))
    num_nodes=len(y)
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (y == i).nonzero().view(-1)
        # index = index[torch.randperm(index.size(0))]
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
