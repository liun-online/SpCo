import argparse
import sys

argv = sys.argv
dataset = argv[1]

def cora_params():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default="cora")    
    parser.add_argument('--gpu', type=int, default=0) 
    parser.add_argument('--hid_units', type=int, default=512)  
    parser.add_argument('--seed', type=int, default=4)
    
    #####################################
    ## new parameters
    parser.add_argument('--delta_origin', type=float, default=0.5)
    parser.add_argument('--sin_iter', type=int, default=3) 
    
    # 20 / 10 labeled nodes per class
    parser.add_argument('--lam', type=float, default=0.1) 
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--turn', type=int, default=20)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--scope_flag', type=int, default=1)
    
    '''
    # 5 labeled nodes per class
    parser.add_argument('--lam', type=float, default=0.1) 
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--turn', type=int, default=30)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--scope_flag', type=int, default=1)
    '''
    #####################################
    
    args, _ = parser.parse_known_args()
    return args
    
def citeseer_params():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default="citeseer")    
    parser.add_argument('--gpu', type=int, default=0) 
    parser.add_argument('--hid_units', type=int, default=512)  
    parser.add_argument('--seed', type=int, default=2)
    
    #####################################
    ## new parameters
    parser.add_argument('--delta_origin', type=float, default=0.5)
    parser.add_argument('--sin_iter', type=int, default=3) 
    
    parser.add_argument('--lam', type=float, default=0.5) 
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--turn', type=int, default=20)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--scope_flag', type=int, default=1)
    #####################################
    
    args, _ = parser.parse_known_args()
    return args
    
def blog_params():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default="blog")    
    parser.add_argument('--gpu', type=int, default=0) 
    parser.add_argument('--hid_units', type=int, default=512)  
    parser.add_argument('--seed', type=int, default=2)
    
    #####################################
    ## new parameters
    parser.add_argument('--delta_origin', type=float, default=0.5)
    parser.add_argument('--sin_iter', type=int, default=3) 
    
    parser.add_argument('--lam', type=float, default=0.3) 
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--turn', type=int, default=30)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--scope_flag', type=int, default=2)
    #####################################
    
    args, _ = parser.parse_known_args()
    return args
    
def flickr_params():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default="flickr")    
    parser.add_argument('--gpu', type=int, default=0) 
    parser.add_argument('--hid_units', type=int, default=512)  
    parser.add_argument('--seed', type=int, default=2)
    
    #####################################
    ## new parameters
    parser.add_argument('--delta_origin', type=float, default=0.5)
    parser.add_argument('--sin_iter', type=int, default=2) 
    
    parser.add_argument('--lam', type=float, default=0.3) 
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--turn', type=int, default=30)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--scope_flag', type=int, default=1)
    #####################################
    
    args, _ = parser.parse_known_args()
    return args
    
def pubmed_params():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default="pubmed")    
    parser.add_argument('--gpu', type=int, default=0) 
    parser.add_argument('--hid_units', type=int, default=256)  
    parser.add_argument('--seed', type=int, default=2)
    
    #####################################
    ## new parameters
    parser.add_argument('--delta_origin', type=float, default=0.5)
    parser.add_argument('--sin_iter', type=int, default=1) 
    
    parser.add_argument('--lam', type=float, default=0.1) 
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--num', type=int, default=3)
    parser.add_argument('--scope_flag', type=int, default=1)
    #####################################
    
    args, _ = parser.parse_known_args()
    return args
    
def set_params():
    if dataset == "cora":
        args = cora_params()
    elif dataset == "citeseer":
        args = citeseer_params()
    elif dataset == "blog":
        args = blog_params()
    elif dataset == "flickr":
        args = flickr_params()
    elif dataset == "pubmed":
        args = pubmed_params()
    return args
