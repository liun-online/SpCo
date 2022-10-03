import numpy as np
import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

from model import Encoder, Model, drop_feature
from eval import label_classification
import scipy.sparse as ssp
from torch_geometric.data import Data
from params import set_params
    
def train(model: Model, x, edge_index, edge_attr, ori_index, ori_attr):
    model.train()
    optimizer.zero_grad()
    edge_index_1, edge_attr_1 = dropout_adj(edge_index, edge_attr, p=drop_edge_rate_1)
    edge_index_2, edge_attr_2 = dropout_adj(ori_index, ori_attr, p=drop_edge_rate_2)
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1, edge_attr_1)
    z2 = model(x_2, edge_index_2, edge_attr_2)
    
    loss = model.loss(z1, z2, batch_size=0)
    if torch.isnan(loss) == True:
        return -1
    loss.backward()
    optimizer.step()

    return loss.item()


def test(own_str, model: Model, x, edge_index, edge_attr, y, idx_train, idx_val, idx_test, final=False):
    model.eval()
    z = model(x, edge_index, edge_attr)
    # z = model(x, edge_index)
    label_classification(own_str, z, y, idx_train, idx_val, idx_test)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = ssp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj, self_loop=False):
    """Symmetrically normalize adjacency matrix."""
    adj = ssp.coo_matrix(adj)
    rowsum = np.array(np.abs(adj.A).sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = ssp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    
def get_dataset(path, dataset, scope_flag):
    adj = ssp.load_npz(path+"/adj.npz").tocoo()
    num_node = adj.shape[0]
    edge = adj.nonzero()
    edge = np.vstack([edge[0], edge[1]])
    edge = torch.LongTensor(edge)
    feat = ssp.load_npz(path+"/feat.npz").A
    if dataset!='blog':
        feat = torch.Tensor(preprocess_features(feat))
    else:
        feat = torch.Tensor(feat)
    num_features = feat.shape[-1]
    label = torch.LongTensor(np.load(path+"/label.npy").astype(np.int32))
    idx_train5 = np.load(path+"/train5.npy").astype(np.int32)
    idx_train10 = np.load(path+"/train10.npy").astype(np.int32)
    idx_train20 = np.load(path+"/train20.npy").astype(np.int32)
    idx_train = [idx_train5, idx_train10, idx_train20]
    idx_val = np.load(path+"/val.npy").astype(np.int32)
    idx_test = np.load(path+"/test.npy").astype(np.int32)
    
    laplace = ssp.eye(adj.shape[0]) - normalize_adj(adj)
    if scope_flag == 1:
        scope = torch.load(path+"/scope_1.pt")
    elif scope_flag == 2:
        scope = torch.load(path+"/scope_2.pt")
    return Data(x=feat, edge_index=edge, y=label), num_node, num_features, idx_train, idx_val, idx_test, adj, laplace, scope


def sinkhorn(K, dist, sin_iter):
    # make the matrix sum to 1
    u = np.ones([len(dist), 1]) / len(dist)
    K_ = ssp.diags(1./dist)*K
    dist = dist.reshape(-1, 1)
    ll = 0
    for it in range(sin_iter):        
        u = 1./K_.dot(dist / (K.T.dot(u)))
    v = dist /(K.T.dot(u))
    delta = np.diag(u.reshape(-1)).dot(K).dot(np.diag(v.reshape(-1)))
    return delta    

def plug(theta, num_node, laplace, delta_add, delta_dele, epsilon, dist, sin_iter, c_flag=False):
    C = (1 - theta)*laplace.A
    if c_flag:
        C = laplace.A
    K_add = np.exp(2 * (C*delta_add).sum() * C / epsilon)
    K_dele = np.exp(-2 * (C*delta_dele).sum() * C / epsilon)
    
    delta_add = sinkhorn(K_add, dist, sin_iter)
    
    delta_dele = sinkhorn(K_dele, dist, sin_iter)
    return delta_add, delta_dele

def update(theta, epoch, total):
    theta = theta - theta*(epoch/total)
    return theta
    
if __name__ == '__main__':
    
    args = set_params()
    own_str = args.dataset
    print(own_str)
    if args.gpu_id!=-1:
        torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]


    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = args.num_epochs
    weight_decay = config['weight_decay']
    
    path = "../dataset/"+args.dataset
    data, num_node, num_features, idx_train, idx_val, idx_test, adj, laplace, scope = get_dataset(path, args.dataset, args.scope_flag)
    if args.dataset!='pubmed':
        scope_matrix = ssp.coo_matrix((np.ones(scope.shape[1]), (scope[0, :], scope[1, :])), shape = adj.shape).A
        dist = adj.A.sum(-1) / adj.A.sum()
    else:
        new_adjs = []
        for i in range(10):
            new_adjs.append(ssp.load_npz(path+"/0.01_1_"+str(i)+".npz"))
        adj_num = len(new_adjs)
        adj_inter = int(adj_num / args.num)
        sele_adjs = []
        for i in range(args.num+1):
            try:
                if i==0:
                    sele_adjs.append(new_adjs[i])
                else:
                    sele_adjs.append(new_adjs[i*adj_inter-1])
            except IndexError:
                pass
        epoch_inter = args.epoch_inter
    
    if args.gpu_id!=-1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
  
    encoder = Encoder(num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers)
    model = Model(encoder, num_hidden, num_proj_hidden, tau)
    if args.gpu_id!=-1:
        encoder = encoder.to(device)
        model = model.to(device)
            
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    #### me ######
    theta = 1
    delta = np.ones(adj.shape) * args.delta_origin
    delta_add = delta
    delta_dele = delta
    
    new_adj = adj.tocsc()
    ori_index = data.edge_index
    ori_attr = torch.Tensor(new_adj[new_adj.nonzero()])[0]
    if args.gpu_id!=-1:
        ori_attr = ori_attr.cuda()
    edge_index = ori_index
    edge_attr = ori_attr
    k = 0
    
    for epoch in range(num_epochs):
        loss = train(model, data.x, edge_index, edge_attr, ori_index, ori_attr)
            
        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now
        
        if args.dataset!='pubmed':
            if epoch % args.turn ==0:
                print("================================================")
                if args.dataset in ["cora", "citeseer"] and epoch!=0:
                    delta_add, delta_dele = plug(theta, num_node, laplace, delta_add, delta_dele, args.epsilon, dist, args.sin_iter, True)
                else:
                    delta_add, delta_dele = plug(theta, num_node, laplace, delta_add, delta_dele, args.epsilon, dist, args.sin_iter)
                delta = normalize_adj((delta_add - delta_dele)* scope_matrix)
                new_adj = adj + args.lam * delta
                
                edge = new_adj.nonzero()
                edge = np.vstack([edge[0], edge[1]])
                edge_index = torch.LongTensor(edge)
                edge_attr = torch.Tensor(new_adj[new_adj.nonzero()])[0]
                theta = update(1, epoch, num_epochs+1)
                if args.gpu_id!=-1:
                    edge_index = edge_index.cuda()
                    edge_attr = edge_attr.cuda()
        else:
            if epoch % epoch_inter == 0 and k<=len(sele_adjs)-1:
                print("================================================")
                delta = args.lam * sele_adjs[k]
                new_adj = adj +  delta
                
                edge = new_adj.nonzero()
                edge = np.vstack([edge[0], edge[1]])
                edge_index = torch.LongTensor(edge)
                edge_attr = torch.Tensor(new_adj[new_adj.nonzero()])[0]
                k+=1
                if args.gpu_id!=-1:
                    edge_index = edge_index.cuda()
                    edge_attr = edge_attr.cuda()
        
        
    '''
    for epoch in range(1, num_epochs + 1):
        loss = train(model, data.x, data.edge_index)

        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now
    '''
    
    print("=== Final ===")
    test(own_str, model, data.x, data.edge_index, ori_attr, data.y, idx_train, idx_val, idx_test, final=True)
