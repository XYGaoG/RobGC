import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import match_loss
import deeprobust.graph.utils as utils
import numpy as np
from models.gcn import GCN
from models.sgc import SGC
from models.sgc_multi import SGC as SGC1
from models.parametrized_adj import PGE
from torch_sparse import SparseTensor
from utils_GSL import *


class GCond_pro:

    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device

        n = int(len(data.idx_train) * args.reduction_rate)
        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))
        self.pge = PGE(nfeat=d, nnodes=n, device=device, args=args).to(device)


        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)
        self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        print('adj_syn:', (n,n), 'feat_syn:', list(self.feat_syn.shape))


        self.knn = args.knn
        self.warm_up = args.warm_up
        self.stru_epoch = args.stru_epoch
        self.paras = [0, 0]
        self.results = []
        self.accs = []
        self.bestdata = []

    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    def generate_labels_syn(self, data):
        from collections import Counter
        counter = Counter(data.labels_train)
        num_class_dict = {}
        n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x:x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}

        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        self.num_class = np.zeros(len(num_class_dict))
        for i in range(len(num_class_dict)):
            self.num_class[i] = self.num_class_dict[i]
        return labels_syn

    def test_with_val(self, verbose=True):
        res = []

        data, device = self.data, self.device
        feat_syn, pge, labels_syn = self.feat_syn.detach(), \
                                self.pge, self.labels_syn
        # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
        dropout = 0.5 if self.args.dataset in ['reddit'] else 0
        model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=dropout,
                    weight_decay=5e-4, nlayers=2,
                    nclass=data.nclass, device=device).to(device)

        adj_syn = pge.inference(feat_syn)
        args = self.args

        if args.save:
            torch.save(adj_syn, f'saved_ours/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
            torch.save(feat_syn, f'saved_ours/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')

        noval = True
        model.fit_with_val(feat_syn, adj_syn, labels_syn, data,
                     train_iters=600, normalize=True, verbose=False, noval=noval)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).to(feat_syn.device)

        output = model.predict(data.feat_test, data.adj_test)

        loss_test = F.nll_loss(output[data.idx_test], labels_test)
        acc_test = utils.accuracy(output[data.idx_test], labels_test)
        res.append(acc_test.item())
        if verbose:
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
        # print(adj_syn.sum().item(), (adj_syn.sum()/(adj_syn.shape[0]**2)).item())

        if False:
            if self.args.dataset == 'ogbn-arxiv':
                thresh = 0.6
            elif self.args.dataset == 'reddit':
                thresh = 0.91
            else:
                thresh = 0.7

            labels_train = torch.LongTensor(data.labels_train).cuda()
            output = model.predict(data.feat_train, data.adj_train)
            # loss_train = F.nll_loss(output, labels_train)
            # acc_train = utils.accuracy(output, labels_train)
            loss_train = torch.tensor(0)
            acc_train = torch.tensor(0)
            if verbose:
                print("Train set results:",
                      "loss= {:.4f}".format(loss_train.item()),
                      "accuracy= {:.4f}".format(acc_train.item()))
            res.append(acc_train.item())
        return res


    def mask_generation(self, label_train, labels_cond):
        classes = torch.unique(label_train)
        mask_real = torch.zeros([len(classes), len(label_train)])
        for cls in classes:
            cls_train_masks= label_train == cls
            mask_real[cls] = cls_train_masks/torch.sum(cls_train_masks)

        mask_real = mask_real.to_sparse()
            
        mask_cond = torch.zeros([len(classes), len(labels_cond)])
        for i, cls in enumerate(classes):
            mask_cond[i, labels_cond == cls] = 1
            mask_cond[i] /= torch.sum(mask_cond[i])
        mask_cond = mask_cond.to_sparse()

        return mask_real.to(self.device), mask_cond.to(self.device)


    def train(self, verbose=True):
        args = self.args
        data = self.data
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn
        features, adj, labels = data.feat_train, data.adj_train, data.labels_train
        syn_class_indices = self.syn_class_indices
        features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        feat_sub, adj_sub = self.get_sub_adj_feat(features)
        self.feat_syn.data.copy_(feat_sub)

        self.data.adj_val_ori = self.data.adj_val
        self.data.adj_test_ori = self.data.adj_test

        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)

        adj_norm = SparseTensor(row=adj_norm._indices()[0], col=adj_norm._indices()[1],
                value=adj_norm._values(), sparse_sizes=adj_norm.size()).t()

        outer_loop, inner_loop = get_loops(args)

        for it in range(args.epochs+1):
            loss_avg = 0
            if args.sgc==1:
                model = SGC(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout,
                            nlayers=args.nlayers, with_bn=False,
                            device=self.device).to(self.device)
            elif args.sgc==2:
                model = SGC1(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout,
                            nlayers=args.nlayers, with_bn=False,
                            device=self.device).to(self.device)

            else:
                model = GCN(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout, nlayers=args.nlayers,
                            device=self.device).to(self.device)

            model.initialize()

            model_parameters = list(model.parameters())

            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
            model.train()

            for ol in range(outer_loop):
                adj_syn = pge(self.feat_syn)
                adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)


                BN_flag = False
                for module in model.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    model.train() # for updating the mu, sigma of BatchNorm
                    output_real = model.forward(features, adj_norm)
                    for module in model.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer

                loss = torch.tensor(0.0).to(self.device)
                for c in range(data.nclass):
                    if c not in self.num_class_dict:
                        continue

                    batch_size, n_id, adjs = data.retrieve_class_sampler(
                            c, adj_norm, transductive=False, args=args)

                    if args.nlayers == 1:
                        adjs = [adjs]
                    adjs = [adj_.to(self.device) for adj_ in adjs]
                    output = model.forward_sampler(features[n_id], adjs)
                    loss_real = F.nll_loss(output, labels[n_id[:batch_size]])
                    gw_real = torch.autograd.grad(loss_real, model_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    ind = syn_class_indices[c]
                    if args.nlayers == 1:
                        adj_syn_norm_list = [adj_syn_norm[ind[0]: ind[1]]]
                    else:
                        adj_syn_norm_list = [adj_syn_norm]*(args.nlayers-1) + \
                                [adj_syn_norm[ind[0]: ind[1]]]

                    output_syn = model.forward_sampler_syn(feat_syn, adj_syn_norm_list)
                    loss_syn = F.nll_loss(output_syn, labels_syn[ind[0]: ind[1]])

                    gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                    coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                    loss += coeff  * match_loss(gw_syn, gw_real, args, device=self.device)

                loss_avg += loss.item()


                # update sythetic graph
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()
                if it % 50 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()

                if args.debug and ol % 5 ==0:
                    print('Gradient matching loss:', loss.item())

                if ol == outer_loop - 1:
                    break


                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn)
                adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=False)
                feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    optimizer_model.step()

            loss_avg /= (data.nclass*outer_loop)
            if it % 50 == 0:
                print('Epoch {},   loss_avg: {:.6f}'.format(it, loss_avg))

            eval_epochs = [100, 200, 400, 800, 1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000]

            #GSL   
            if (it == self.warm_up) or (it >=self.warm_up and it % self.stru_epoch == 0):
                _, adj_norm = GSL(features, adj, labels, feat_syn, pge, self.paras, self.knn)

            if verbose and (it in eval_epochs or it == args.epochs):
                #GSL
                if it >=self.warm_up:
                    print('____________GSL eval graph:____________')
                    self.data.adj_val  = GSL_inf(self.data.feat_val,  self.data.adj_val_ori,  feat_syn, pge, self.paras, self.knn)
                    self.data.adj_test = GSL_inf(self.data.feat_test, self.data.adj_test_ori, feat_syn, pge, self.paras, self.knn)

                res = []
                runs = 3
                for _ in range(runs):
                    res.append(self.test_with_val())
                res = np.array(res)
                self.accs.append(round(res.mean(0)[0].item()*100, 2)) 
                print('Test:', [round(res.mean(0)[0].item()*100, 2), round(res.std(0)[0].item()*100, 2)])
                self.results.append(res)
                adj_syn = pge.inference(feat_syn)
                paras = [self.paras[0], self.paras[1]]
                self.bestdata.append([feat_syn.detach(),adj_syn.detach(),labels_syn,paras])
        idx = np.argmax(self.accs)
        res = self.results[idx]
        best_data = self.bestdata[idx]
        if self.args.cross_save_data == True:
            torch.save(best_data, f'./res/{self.args.dataset}_{self.args.GC}_pro_random_1_mid.pth')
        return res

    def get_sub_adj_feat(self, features):
        data = self.data
        args = self.args
        idx_selected = []

        from collections import Counter;
        counter = Counter(self.labels_syn.cpu().numpy())

        for c in range(data.nclass):
            tmp = data.retrieve_class(c, num=counter[c])
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[idx_selected]

        # adj_knn = torch.zeros((data.nclass*args.nsamples, data.nclass*args.nsamples)).to(self.device)
        # for i in range(data.nclass):
        #     idx = np.arange(i*args.nsamples, i*args.nsamples+args.nsamples)
        #     adj_knn[np.ix_(idx, idx)] = 1

        from sklearn.metrics.pairwise import cosine_similarity
        # features[features!=0] = 1
        k = 2
        sims = cosine_similarity(features.cpu().numpy())
        sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
        for i in range(len(sims)):
            indices_argsort = np.argsort(sims[i])
            sims[i, indices_argsort[: -k]] = 0
        adj_knn = torch.FloatTensor(sims).to(self.device)
        return features, adj_knn



def get_loops(args):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if args.one_step:
        return 10, 0

    if args.dataset in ['ogbn-arxiv']:
        return 20, 0
    if args.dataset in ['reddit']:
        return args.outer, args.inner
    if args.dataset in ['flickr']:
        return 20, 3
        # return 10, 1
    if args.dataset in ['cora']:
        return 20, 10
    if args.dataset in ['citeseer']:
        return 20, 5 # at least 200 epochs
    else:
        return 20, 5



