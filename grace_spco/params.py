import argparse
import sys

argv = sys.argv
dataset = argv[1]

def cora_params():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default="cora")    
    parser.add_argument('--gpu_id', type=int, default=0) 
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--seed', type=int, default=4) 
    
    #####################################
    ## new parameters
    parser.add_argument('--delta_origin', type=float, default=0.5)
    parser.add_argument('--theta', type=float, default=1.)  
    parser.add_argument('--sin_iter', type=int, default=3) 
    
    parser.add_argument('--lam', type=float, default=0.5) 
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--turn', type=int, default=30)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--scope_flag', type=int, default=1)
    #####################################
    
    args, _ = parser.parse_known_args()
    return args
    
def citeseer_params():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default="citeseer") 
    parser.add_argument('--gpu_id', type=int, default=0) 
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--seed', type=int, default=0) 
    
    #####################################
    ## new parameters
    parser.add_argument('--delta_origin', type=float, default=0.5)
    parser.add_argument('--theta', type=float, default=1.)  
    parser.add_argument('--sin_iter', type=int, default=3) 
    
    parser.add_argument('--lam', type=float, default=1.0) 
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--turn', type=int, default=20)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--scope_flag', type=int, default=1)
    #####################################
    
    args, _ = parser.parse_known_args()
    return args
    
def blog_params():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default="blog") 
    parser.add_argument('--gpu_id', type=int, default=0) 
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--seed', type=int, default=0) 
    
    #####################################
    ## new parameters
    parser.add_argument('--delta_origin', type=float, default=0.5)
    parser.add_argument('--sin_iter', type=int, default=3) 
    
    parser.add_argument('--lam', type=float, default=1.0) 
    parser.add_argument('--num_epochs', type=int, default=800)
    parser.add_argument('--turn', type=int, default=300)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--scope_flag', type=int, default=1)
    #####################################
    
    args, _ = parser.parse_known_args()
    return args
    
def flickr_params():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default="flickr") 
    parser.add_argument('--gpu_id', type=int, default=0) 
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--seed', type=int, default=0) 
    
    #####################################
    ## new parameters
    parser.add_argument('--delta_origin', type=float, default=0.5)
    parser.add_argument('--theta', type=float, default=1.)  
    parser.add_argument('--sin_iter', type=int, default=2) 
    
    parser.add_argument('--lam', type=float, default=1.0) 
    parser.add_argument('--num_epochs', type=int, default=1300)
    parser.add_argument('--turn', type=int, default=300)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--scope_flag', type=int, default=1)
    #####################################
    
    args, _ = parser.parse_known_args()
    return args

def pubmed_params():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default="pubmed")   
    parser.add_argument('--gpu_id', type=int, default=0) 
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--seed', type=int, default=0) 
    
    #####################################
    ## new parameters
    parser.add_argument('--delta_origin', type=float, default=0.5)
    parser.add_argument('--theta', type=float, default=1.)  
    parser.add_argument('--sin_iter', type=int, default=1) 
    
    parser.add_argument('--lam', type=float, default=1.0) 
    parser.add_argument('--num_epochs', type=int, default=1500)
    parser.add_argument('--num', type=int, default=7)
    parser.add_argument('--epoch_inter', type=int, default=100)
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
