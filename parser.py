# -*- coding: utf-8 -*-
import argparse
import warnings

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2

warnings.filterwarnings("ignore")

def get_parser():
    parser = argparse.ArgumentParser()
    # General settings
    parser.add_argument('--resume', action='store_true', dest='resume', default=False)
    parser.add_argument('--dataset', dest='dataset', default='example') 
    parser.add_argument('--gpu', dest='gpu', type=bool, default=False) 
    
    # Network settings
    parser.add_argument('--n_features', dest='n_features', type=int, default=128)
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=128)
    
    # Training settings
    parser.add_argument('--epochs', dest='epochs', type=int, default=2)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)   
    parser.add_argument("--optim", dest="optim", default='Adam')
    parser.add_argument('--lr', dest='lr', default=0.003) 

    parser.add_argument('--weight_decay', type=float, default=5e-6) 
    parser.add_argument('--neg', dest='neg', type=int, default=2)  
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.6) 
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.2)
    parser.add_argument('--beam_size', dest='beam_size', type=int, default=2)  
    parser.add_argument('--nheads', dest='nheads', type=int, default=4)
    
    parser.add_argument("--n_ctxs", dest="n_ctxs", type=int, default=2) 
    parser.add_argument("--ctx_attn", dest="ctx_attn", type=str, default='sum_co') 
    
    # Multi-View multi-task settings
    parser.add_argument("--integration", dest="integration",  type=str, default='')
    parser.add_argument("--lambda", dest="lamb", type=float, default=0.05) 
    parser.add_argument("--task", dest="task",  type=str, default='multi')
    
    opt = parser.parse_args()
    return opt

