import numpy as np
import random
import argparse
import torch
from utils import *
import os
from scipy.sparse import save_npz
from torch_geometric.utils.dropout import dropout_edge
from torch_geometric.utils.augmentation import add_random_edge
from torch_geometric.utils import coalesce, to_scipy_sparse_matrix



parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=-1, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--dis_metric', type=str, default='ours')
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--nlayers', type=int, default=3)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--alpha', type=float, default=0, help='regularization term.')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--option', type=int, default=0)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--label_rate', type=float, default=1)
parser.add_argument('--one_step', type=int, default=0)


parser.add_argument('--attack', type=str, default='random') 
parser.add_argument('--ptb_rate', type=float, default=1., help="noise ptb_rate")

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if args.ptb_rate == 0:
    args.attack = "no"

data_full = get_dataset(args.dataset, args.normalize_features)
data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)

adj = data.adj_full
print('#edges in adj:', data.adj_full.sum())
if args.attack == 'no':
    perturbed_adj = adj
    adj = data.adj_full
    print('#edges in adj:', data.adj_full.sum())

if args.attack == 'random':
    adj = data.adj_full
    print('#edges in adj:', data.adj_full.sum())
    row, col = adj.nonzero()
    adj = torch.stack([torch.LongTensor(row), torch.LongTensor(col)], dim=0)
    n_perturbations = int(args.ptb_rate * (adj.shape[1]))
    sparse = adj.shape[1]/(data.adj_full.shape[0]*data.adj_full.shape[0])
    p_decrease = n_perturbations*sparse/adj.shape[1]
    adj, _ = dropout_edge(adj, force_undirected=True, p=p_decrease)
    p_increase = min(n_perturbations*(1-sparse)/adj.shape[1],1.)
    adj, _ = add_random_edge(adj, force_undirected=True, p=p_increase, num_nodes=data.adj_full.shape[0])     
    adj = coalesce(adj, num_nodes=None, is_sorted=False)
    adj = to_undirected(adj)
    adj = to_scipy_sparse_matrix(adj, num_nodes  = data.adj_full.shape[0])
    perturbed_adj = adj.tocsr()

print('#edges in adj:', perturbed_adj.sum())
folder_path = f"./attacked_adj/{args.dataset}/"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
save_npz(folder_path+f'adj_{args.attack}_{args.ptb_rate}.npz', perturbed_adj)
