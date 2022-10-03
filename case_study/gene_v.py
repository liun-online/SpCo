import numpy as np
import scipy.sparse as sp

dataset="flickr"

#### The eigenvalue decomposition of target adjacency matrix ####
adj = sp.load_npz("../dataset/"+dataset+"/adj.npz").A.astype(np.int32)
num = adj.shape[0]
degree = np.diag(adj.sum(-1)**(-0.5))
a_ = degree.dot(adj.dot(degree))
lap = np.eye(num)-a_

va, ve = np.linalg.eig(lap)
ll = ve.dot(ve.T)

index = np.argsort(va)
va = va[index]
np.save("../dataset/"+dataset+"/va_"+dataset+".npy",va)
ve = ve[:, index]
np.save("../dataset/"+dataset+"/ve_"+dataset+".npy",ve)

#### Generate V ####
ratio = [0.2,0.4,0.6,0.8]
interval = int(num / 2)

low_base = ve[:, :interval].dot(ve[:, :interval].T)
low_len = interval
high_base = ve[:, interval:].dot(ve[:, interval:].T)
high_len = num - interval

sp.save_npz("../dataset/"+dataset+"/"+dataset+"low0.npz", sp.coo_matrix(high_base))
sp.save_npz("../dataset/"+dataset+"/"+dataset+"hig0.npz", sp.coo_matrix(low_base))
for i in ratio:
    l_low = int(low_len*i)
    low = ve[:, :l_low]
    l_hig = int(high_len*i)
    hig = ve[:, interval: interval+l_hig]
    print(low.shape,hig.shape)
    
    low = sp.coo_matrix(low.dot(low.T)+high_base)
    hig = sp.coo_matrix(hig.dot(hig.T)+low_base)
    sp.save_npz("../dataset/"+dataset+"/"+dataset+"low"+str(i)+".npz", low)
    sp.save_npz("../dataset/"+dataset+"/"+dataset+"hig"+str(i)+".npz", hig)