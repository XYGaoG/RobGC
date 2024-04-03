import numpy as np
import random
import argparse
import torch
from utils import *
from gcond_agent_induct_pro import GCond_pro


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=-1, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--dis_metric', type=str, default='mse')
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--lr_adj', type=float, default=0.001)
parser.add_argument('--lr_feat', type=float, default=0.001)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=0.058)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--sgc', type=int, default=0)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--option', type=int, default=0)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--label_rate', type=float, default=1)
parser.add_argument('--one_step', type=int, default=1)

parser.add_argument('--attack', type=str, default='random')  
parser.add_argument('--ptb_rate', type=float, default=1., help="noise level")
parser.add_argument('--GC', type=str, default='gcond')
parser.add_argument('--method', type=str, default='proposed')

parser.add_argument('--warm_up', type=float, default=50)
parser.add_argument('--stru_epoch', type=float, default=50)
parser.add_argument('--knn', type=int, default=2)
parser.add_argument('--cross_save_data', type=bool, default=False)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.gpu_id)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
args.ptb_rate = np.float64(args.ptb_rate)
print(args)

data_full = get_dataset(args.dataset, args.normalize_features)
data = Transd2Ind_attack(data_full, keep_ratio=args.keep_ratio, attack=args.attack, ptb_rate=args.ptb_rate, dataset=args.dataset, method = args.method)

agent = GCond_pro(data, args, device=device)
res = agent.train()
result_path = f"./{args.GC}/{args.dataset}/{args.reduction_rate}/{args.attack}/"
method_name = f"{args.method}"
result_record_acc_pro(args.ptb_rate, res, result_path, method_name, args)