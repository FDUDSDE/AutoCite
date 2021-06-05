import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2

class HistDataset(Dataset):
    def __init__(self, loader, opt, train = True):
        self.df = loader.train_data
        self.word2idx = loader.word2idx
        self.G_out = loader.G_out
        self.G_in = loader.G_in
        self.n_ctxs = int(opt.n_ctxs) 
        self.neg = opt.neg
        self.init()
        if train:
            self.data = self.df.values 
        else:
            self.data = loader.test_data.values
        
       
    def ctx2words(self, context):
        return [self.word2idx[str(word)] for word in context.split()]


    def init(self):
        self.heads = set(self.df['paper_id1'])
        self.tails =set(self.df['paper_id2']) 

        self.head_ctx = {}
        self.tail_ctx = {}
        for h in set(self.heads):
            self.head_ctx[h] = [c['context'] for _,_,c in self.G_out.edges([h], data = True)]
  
        for t in set(self.tails):
            self.tail_ctx[t] = [c['context'] for _,_,c in self.G_in.edges([t], data = True)]
      
        
    def neg_triples(self, h_id, t_id):
        all_heads = self.heads
        all_tails = self.tails
        neg = []
   
        ts = all_tails - set(self.G_out.neighbors(h_id))
        hs = all_heads - set(self.G_in.neighbors(t_id))
        
        # samples
        chose = np.random.randint(0, 2, self.neg)
        h_neg = np.sum(chose == 0)
        t_neg = np.sum(chose == 1)
        neg_tails = random.sample(ts, h_neg)
        neg_heads = random.sample(hs, t_neg)

        # construct neg triples
        neg_h_id = neg_heads
        neg_t_id = [t_id]*len(neg_heads)
        
        neg_t_id += neg_tails
        neg_h_id += [h_id]*len(neg_tails)
        return neg_h_id, neg_t_id
        
    
    def __len__(self):
        return len(self.data)
    
    
    def h_context(self, h_id):
        h_ctxs = np.random.choice(self.head_ctx[h_id], self.n_ctxs)
        h_ctxs = [self.ctx2words(ctx) for ctx in h_ctxs]
        return h_ctxs
    
    def t_context(self, t_id):
        t_ctxs = np.random.choice(self.tail_ctx[t_id], self.n_ctxs)
        t_ctxs = [self.ctx2words(ctx) for ctx in t_ctxs]
        return t_ctxs
    
        
    def __getitem__(self, idx):
        h_id, t_id, clean_content, num_of_words = self.data[idx]        
        content_idx = [self.word2idx[str(word)] for word in clean_content.split()]
        neg_h_id, neg_t_id = self.neg_triples(h_id, t_id)  
        
        #word_input = [SOS_TOKEN] + content_idx
        word_input = content_idx
        word_output = content_idx + [EOS_TOKEN]
        
        h_ctxs = self.h_context(h_id)
        t_ctxs = self.t_context(t_id)
        h_ctx_len = [len(h_ctx) for h_ctx in h_ctxs]
        t_ctx_len = [len(t_ctx) for t_ctx in t_ctxs]
        
        neg_h_ctxs = [ctx for neg_h in neg_h_id for ctx in self.h_context(neg_h)]
        neg_t_ctxs = [ctx for neg_t in neg_t_id for ctx in self.t_context(neg_t)]
        neg_h_ctx_len = [len(h_ctx) for h_ctx in neg_h_ctxs]
        neg_t_ctx_len = [len(t_ctx) for t_ctx in neg_t_ctxs]
            
        return h_id, t_id, neg_h_id, neg_t_id, \
                h_ctxs, t_ctxs, h_ctx_len, t_ctx_len, \
                neg_h_ctxs, neg_t_ctxs, neg_h_ctx_len, neg_t_ctx_len, \
                word_input, word_output, num_of_words+1

