import numpy as np
import scipy.sparse as sp
import networkx as nx

############################################################
# Please notice that other existing augmentations can be obtained by corresponding methods. 
# Here, we only give the generation of heat matrix and distance matrix.
# For the heat matrix, the authors give the original code. But we notice that in the given code, 
# the authors falsely operate exp(·) in an element-wise way, rather than as a matrix exponential operation. Here, we correct the mistake.
# For the distance matrix, we obey the instruction of the original paper.
############################################################

##### Heat matrix ####
dataset="cora"
adj = sp.load_npz("../dataset/"+dataset+"/adj.npz").A.astype(np.int32)
num = adj.shape[0]
d = np.diag(np.sum(adj, 1))
a_ = np.matmul(adj, inv(d))
lap = np.eye(num)-a_

va, ve = np.linalg.eig(lap)
index = np.argsort(va)
va = va[index]
ve = ve[:, index]

ll = (-va)*5 ## t=5
ll = np.exp(ll)
ll = sp.coo_matrix(ve.dot(np.diag(ll).dot(ve.T)))
sp.save_npz("../dataset/"+dataset+"/"+dataset+"_heat.npz", ll)

##### Distance matrix ####
G = nx.from_scipy_sparse_matrix(adj)
ll = nx.floyd_warshall(G)
mm = np.zeros(adj.shape)
for key,value in ll.items():
    for k,v in value.items():
        mm[key][k] = 1 / v
mm = sp.coo_matrix(mm)
sp.save_npz("../dataset/"+dataset+"/"+dataset+"_dis.npz", mm)
