import numpy as np
import torch
import torch.nn.functional as F
import deeprobust.graph.utils as utils
import numpy as np
import scipy.sparse as sp
import time
import torch_sparse



def label_propagation_multi(adj, labels):
    a = 0.1
    h = labels
    for _ in range(10):
        h = (1-a)*torch.sparse.mm(adj, h) + a*labels  
    return F.log_softmax(h, dim=1)


def normalize_adj_sparse(mx):
    rowsum = torch.sparse.sum(mx, dim=1)
    r_inv = rowsum.pow(-1/2).flatten().to_dense()
    r_inv[torch.isinf(r_inv)] = 0.
    # r_mat_inv = torch.diag(r_inv)
    indices = torch.arange(r_inv.size(0))
    r_indices = torch.stack([indices, indices]).to(mx.device)
    mx_value = mx._values()
    mx_indices = mx._indices()
    idx, val = torch_sparse.spspmm(r_indices, r_inv, mx_indices, mx_value, mx.shape[0], mx.shape[0], mx.shape[0])
    idx, val = torch_sparse.spspmm(idx, val, r_indices, r_inv, mx.shape[0], mx.shape[0], mx.shape[0])
    mx = torch.sparse.FloatTensor(idx, val, torch.Size([mx.shape[0], mx.shape[0]]))
    return mx


def normalize_sparse_tensor(self, adj, fill_value=1):
    edge_index = adj._indices()
    edge_weight = adj._values()
    num_nodes= adj.size(0)
    edge_index, edge_weight = self.add_self_loops( 
    edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    from torch_scatter import scatter_add
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    shape = adj.shape
    return torch.sparse.FloatTensor(edge_index, values, shape)





@torch.no_grad()
def GSL(features, adj, labels, feat_syn, pge, paras, knn):
    if sp.issparse(adj):
        adj = utils.sparse_mx_to_torch_sparse_tensor(adj).to(features.device)
    elif type(adj) is not torch.Tensor:
        adj = torch.FloatTensor(adj).to(features.device)
    if isinstance(features, np.ndarray):
        features = torch.FloatTensor(features)

    step_num=20
        
    adj_syn = pge.inference(feat_syn)
    adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)

    labels_one_hot = utils.tensor2onehot(labels.to('cpu')).to(features.device)
    idx = torch.randperm(len(labels))[:int(len(labels)*0.5)]
    labels_one_hot[idx,:] *= 0

    embeddings_real = F.normalize(features, p=2, dim=-1)
    embeddings_syn  = F.normalize(feat_syn, p=2, dim=-1)
    E  = torch.matmul(embeddings_real, embeddings_syn.transpose(-1, -2)).t()
    E1 = torch.matmul(adj_syn_norm, E)
    E2 = torch.matmul(adj_syn_norm, E1)

    M = torch.concat((E, E, E), dim=0).t()
    # M = torch.concat((features.t(), E, E, E), dim=0).t()
    M = F.normalize(M, p=2, dim=1)  

    Mt = torch.concat((E, E1, E2), dim=0).t()
    # Mt = torch.concat((features.t(), E, E1, E2), dim=0).t()
    Mt = F.normalize(Mt, p=2, dim=1)

    # delete edge
    rows_1 = adj._indices()[0]
    cols_1 = adj._indices()[1]
    M_row = M[rows_1,:].unsqueeze(1)
    M_col = Mt[cols_1,:].unsqueeze(2)
    values_1 = (M_row @ M_col).squeeze()

    # add edge        
    rows, cols, values = knn_fast(M, Mt.t(), knn, 1000)
    rows_2 = torch.cat((rows, cols))
    cols_2 = torch.cat((cols, rows))
    values_2 = torch.cat((values, values))

    min1= values_1.min().item()
    max1= values_1.max().item()
    min2= values_2.min().item()
    max2= values_2.max().item()

    search_time_begin = time.time()
    acc_list=[]
    paras_list=[]
    be_time=time.time()
    for c1 in np.linspace(min1, max1, step_num, endpoint=True):
        for c2 in np.linspace(min2, max2, step_num, endpoint=True):
            mask_1 = values_1 > c1        
            mask_2 = values_2 > c2
            r = torch.cat((rows_1[mask_1], rows_2[mask_2]))
            c = torch.cat((cols_1[mask_1], cols_2[mask_2]))
            v = torch.ones(mask_1.sum().item() + mask_2.sum().item()).to(features.device)
            adj_new = torch.sparse.FloatTensor(torch.stack([r, c], dim=0), v, torch.Size([features.shape[0], features.shape[0]]))
            adj_new = adj_new.coalesce()
            adj_norm = normalize_adj_sparse(adj_new)
            output = label_propagation_multi(adj_norm, labels_one_hot)
            acc = utils.accuracy(output, labels)
            acc_list.append(acc.item())
            paras_list.append([c1, c2]) 
    print(f'search time: {time.time()-search_time_begin:.2f}')


    paras[0] = paras_list[np.argmax(acc_list)][0]
    paras[1] = paras_list[np.argmax(acc_list)][1]

    mask_1 = values_1 > paras[0]       
    mask_2 = values_2 > paras[1]
    r = torch.cat((rows_1[mask_1], rows_2[mask_2]))
    c = torch.cat((cols_1[mask_1], cols_2[mask_2]))
    v = torch.ones(mask_1.sum().item() + mask_2.sum().item()).to(features.device)
    adj_new = torch.sparse.FloatTensor(torch.stack([r, c], dim=0), v, torch.Size([features.shape[0], features.shape[0]]))
    adj_new = adj_new.coalesce()
    adj_norm = normalize_adj_sparse(adj_new)  
    print(f'GSL train edges: {adj._values().shape[0]} >>> {adj_new._values().shape[0]},  time: {time.time()-be_time:.2f}')
    return adj_new, adj_norm


@torch.no_grad()
def GSL_inf(features, adj, feat_syn, pge, paras, knn):
    flag_sp = 1 if sp.issparse(adj) else 0
    if sp.issparse(adj):
        adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
    elif type(adj) is not torch.Tensor:
        adj = torch.FloatTensor(adj)
    if isinstance(features, np.ndarray):
        features = torch.FloatTensor(features)

    adj_syn = pge.inference(feat_syn)
    adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False).to('cpu')

    embeddings_real = F.normalize(features.to('cpu'), p=2, dim=-1)
    embeddings_syn  = F.normalize(feat_syn.to('cpu'), p=2, dim=-1)
    E  = torch.matmul(embeddings_real, embeddings_syn.transpose(-1, -2)).t()
    E1 = torch.matmul(adj_syn_norm, E)
    E2 = torch.matmul(adj_syn_norm, E1)

    M = torch.concat((E, E, E), dim=0).t()
    # M = torch.concat((features.t(), E, E, E), dim=0).t()
    M = F.normalize(M, p=2, dim=1)  

    Mt = torch.concat((E, E1, E2), dim=0).t()
    # Mt = torch.concat((features.t(), E, E1, E2), dim=0).t()
    Mt = F.normalize(Mt, p=2, dim=1)

    # delete edge
    rows_1 = adj._indices()[0]
    cols_1 = adj._indices()[1]
    M_row = M[rows_1,:].unsqueeze(1)
    M_col = Mt[cols_1,:].unsqueeze(2)
    values_1 = (M_row @ M_col).squeeze()
    # add edge        
    rows, cols, values = knn_fast(M, Mt.t(), knn, 1000)
    rows_2 = torch.cat((rows, cols))
    cols_2 = torch.cat((cols, rows))
    values_2 = torch.cat((values, values))

    mask_1 = values_1 > paras[0]       
    mask_2 = values_2 > paras[1]
    r = torch.cat((rows_1[mask_1], rows_2[mask_2]))
    c = torch.cat((cols_1[mask_1], cols_2[mask_2]))
    v = torch.ones(mask_1.sum().item() + mask_2.sum().item()).to(features.device)
    adj_new = torch.sparse.FloatTensor(torch.stack([r, c], dim=0), v, torch.Size([features.shape[0], features.shape[0]]))
    adj_new = adj_new.coalesce()
    print(f'GSL eval edges: {adj._values().shape[0]} >>> {adj_new._values().shape[0]}')

    if flag_sp==1:
        adj_new = sp.coo_matrix((adj_new._values().cpu().numpy(), 
        (adj_new._indices()[0].cpu().numpy(), adj_new._indices()[1].cpu().numpy())), 
        shape=adj_new.size()).tocsr()
    return adj_new

def knn_fast(X, Y, k, b):
    device = X.device
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).to(device)
    rows = torch.zeros(X.shape[0] * (k + 1)).to(device)
    cols = torch.zeros(X.shape[0] * (k + 1)).to(device)
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, Y)
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        index += b
    rows = rows.long()
    cols = cols.long()
    return rows, cols, values
