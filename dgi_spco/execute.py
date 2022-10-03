import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import os
from models import DGI, LogReg
from utils import process
import argparse
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score
from params import set_params

args = set_params()
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")
'''
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
'''

own_str = args.dataset
print(own_str)

pkl_name = own_str+".pkl"

def sinkhorn(K, dist, sin_iter):
    # make the matrix sum to 1
    u = np.ones([len(dist), 1]) / len(dist)
    K_ = sp.diags(1./dist)*K
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
    
def gen_auc_mima(logits, label):
    preds = torch.argmax(logits, dim=1)
    test_f1_macro = f1_score(label.cpu(), preds.cpu(), average='macro')
    test_f1_micro = f1_score(label.cpu(), preds.cpu(), average='micro')
    
    best_proba = F.softmax(logits, dim=1)
    if logits.shape[1] != 2:
        auc = roc_auc_score(y_true=label.detach().cpu().numpy(),
                                                y_score=best_proba.detach().cpu().numpy(),
                                                multi_class='ovr'
                                                )
    else:
        auc = roc_auc_score(y_true=label.detach().cpu().numpy(),
                                                y_score=best_proba[:,1].detach().cpu().numpy()
                                                )
    return test_f1_macro, test_f1_micro, auc

# training params
dataset = args.dataset
batch_size = 1
nb_epochs = 10000
patience = args.patience
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = args.hid_units
  
sparse = True
nonlinearity = 'prelu' # special name to separate parameters

adj_ori, features, labels, idx_train, idx_val, idx_test, laplace, scope = process.load_data(args.dataset, args.scope_flag)
num_node = adj_ori.shape[0]
if args.dataset == 'pubmed':
    new_adjs = []
    for i in range(10):
        new_adjs.append(sp.load_npz("../dataset/pubmed/0.01_1_"+str(i)+".npz"))
    adj_num = len(new_adjs)
    adj_inter = int(adj_num / args.num)
    sele_adjs = [new_adjs[i*adj_inter] for i in range(args.num+1)]
    epoch_inter = int(300 / (args.num+1))
else:
    scope_matrix = sp.coo_matrix((np.ones(scope.shape[1]), (scope[0, :], scope[1, :])), shape = adj_ori.shape).A
    dist = adj_ori.A.sum(-1) / adj_ori.A.sum()

if args.dataset != 'blog' :
    features,_ = process.preprocess_features(features)
else:
    features = features.todense()

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]

adj = process.normalize_adj(adj_ori + sp.eye(adj_ori.shape[0]))
  
if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
idx_train = [torch.LongTensor(i) for i in idx_train]
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

model = DGI(ft_size, hid_units, nonlinearity)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
    else:
        adj = adj.cuda()
    labels = labels.cuda()
    idx_train = [i.cuda() for i in idx_train] 
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0

#### me ######
theta = 1
delta = np.ones(adj.shape) * args.delta_origin
delta_add = delta
delta_dele = delta
new_adj = sp_adj
for epoch in range(nb_epochs):
    model.train()
    optimiser.zero_grad()

    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]

    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1)
    
    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()
    
    logits = model(features, shuf_fts, new_adj, sp_adj if sparse else adj, sparse, None, None, None) 

    loss = b_xent(logits, lbl)
    if torch.isnan(loss) == True:
        break
    print('Loss:', loss)

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), pkl_name)
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()
    if args.dataset == 'pubmed':
        if (epoch-1) % epoch_inter == 0:
            if epoch > 300:
                pass
            else:
                print("================================================")
                delta = args.lam *sele_adjs[int(epoch / epoch_inter)]
                delta = process.sparse_mx_to_torch_sparse_tensor(delta).cuda()
                new_adj = sp_adj +  delta 
    else:
        if epoch % args.turn ==0:
            print("================================================")
            if args.dataset in ["cora", "citeseer"] and epoch!=0:
                delta_add, delta_dele = plug(theta, num_node, laplace, delta_add, delta_dele, args.epsilon, dist, args.sin_iter, True)
            else:
                delta_add, delta_dele = plug(theta, num_node, laplace, delta_add, delta_dele, args.epsilon, dist, args.sin_iter)
            delta = (delta_add - delta_dele)* scope_matrix
            delta = args.lam * process.normalize_adj(delta)
           
            delta = process.sparse_mx_to_torch_sparse_tensor(delta).cuda()
            new_adj = sp_adj +  delta
            theta = update(1, epoch, 200)    
    

print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load(pkl_name))

embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
os.remove(pkl_name)

label_dict = {0:"5", 1:"10", 2:"20"}
for i in range(3):
    train_embs = embeds[0, idx_train[i]]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]
    
    train_lbls = torch.argmax(labels[0, idx_train[i]], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)
    
    tot = torch.zeros(1)
    tot = tot.cuda()
    
    auc = []
    test_f1_micro = []
    test_f1_macro = []
    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.cuda()
    
        pat_steps = 0
        best_acc = torch.zeros(1)
        best_acc = best_acc.cuda()
        for _ in range(100):
            log.train()
            opt.zero_grad()
    
            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            
            loss.backward()
            opt.step()
    
        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        test_f1_macro.append(f1_score(test_lbls.cpu(), preds.cpu(), average='macro'))
        test_f1_micro.append(f1_score(test_lbls.cpu(), preds.cpu(), average='micro'))
    test_f1_micro = np.array(test_f1_micro).mean()
    test_f1_macro = np.array(test_f1_macro).mean()
    f=open(own_str+"_"+label_dict[i]+".txt","a")
    f.write(str(test_f1_macro)+"\t"+str(test_f1_micro)+"\n")
    f.close()