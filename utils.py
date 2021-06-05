import time
import pickle
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2


def compute_bleu(references, candidates):
    references = [[item] for item in references]
    smooth_func = SmoothingFunction().method0  # method0-method7
    score1 = corpus_bleu(references, candidates, smoothing_function=smooth_func, weights=(1.0,))
    score2 = corpus_bleu(references, candidates, smoothing_function=smooth_func, weights=(1.0/2,)*2)
    score3 = corpus_bleu(references, candidates, smoothing_function=smooth_func, weights=(1.0/3,)*3)
    score4 = corpus_bleu(references, candidates, smoothing_function=smooth_func, weights=(1.0/4,)*4)
    return score1, score2, score3, score4


class Loader():
    def __init__(self, dataset, gpu):
        self.word2idx = {}
        self.idx2word = {0: "SOS", 
                         1: "EOS", 
                         2: "PAD"}
        self.n_words = len(self.idx2word) 
        self.G_out = nx.DiGraph()
        self.G_in = nx.DiGraph()
        
        self.train_data, self.test_data = self.load_data(dataset)
        self.cons_vocabulary(self.train_data)
        self.cons_vocabulary(self.test_data)
        self.cons_graph(self.train_data)
        self.gpu = gpu
       
    
    def load_data(self, dataset):
        path = 'data/{}.pkl'.format(dataset)
        train_data, test_data = pickle.load(open(path, 'rb'))

        self.max_len = train_data["num_of_words"].max()
        self.n_nodes = max(train_data["paper_id1"].max(), train_data["paper_id2"].max())+1
        
        return train_data, test_data

    
    def build_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1  
            
            
    def cons_vocabulary(self, data):
        for index, row in data.iterrows():
            for word in row["clean_content"].split():
                self.build_word(word)
        
        
    def cons_graph(self, data):
        for p1, p2, c, _ in data.values:
            self.G_out.add_edge(p1, p2, context=c) 
            self.G_in.add_edge(p2, p1, context=c) 
         
    
    def subgraph(self, idx):  
        # in_graph
        in_sub_nodes = set()
        for node in (idx):
            in_sub_nodes.add(node)
            neibors = self.G_in.neighbors(node)
            in_sub_nodes = in_sub_nodes.union(set(neibors))    
        in_sub_graph = nx.subgraph(self.G_in, list(in_sub_nodes))
     
        # out_graph
        out_sub_nodes = set()
        for node in (idx):
            out_sub_nodes.add(node)
            neibors = self.G_out.neighbors(node)
            out_sub_nodes = out_sub_nodes.union(set(neibors))
        out_sub_graph = nx.subgraph(self.G_out, list(out_sub_nodes))
        return in_sub_graph, out_sub_graph
    
    
    def graph_subtensor(self, hs, ts):         
        in_sub_graph, out_sub_graph = self.subgraph(hs+ts)
        
        in_nodes = list(in_sub_graph.nodes())
        in_adj =  np.array(nx.adjacency_matrix(in_sub_graph).todense())
        out_nodes = list(out_sub_graph.nodes())        
        out_adj = np.array(nx.adjacency_matrix(out_sub_graph).todense())
     
        # node mapping 
        in_node2idx = {n:i for i, n in enumerate(in_nodes)}
        out_node2idx = {n:i for i, n in enumerate(out_nodes)}
        
        in_map_hs = [in_node2idx[i] for i in hs]
        in_map_ts = [in_node2idx[i] for i in ts]
        out_map_hs = [out_node2idx[i] for i in hs]
        out_map_ts = [out_node2idx[i] for i in ts]
        
        # convert to tensors
        in_data = [torch.LongTensor(in_nodes), torch.FloatTensor(in_adj),
                   torch.LongTensor(in_map_hs), torch.LongTensor(in_map_ts)]
        out_data = [torch.LongTensor(out_nodes), torch.FloatTensor(out_adj),
                   torch.LongTensor(out_map_hs), torch.LongTensor(out_map_ts)]
        
        if self.gpu:
            hs = torch.LongTensor(hs).cuda()
            ts = torch.LongTensor(ts).cuda()
            in_data = [x.cuda() for x in in_data]
            out_data = [x.cuda() for x in out_data]
        else:
            hs = torch.LongTensor(hs)
            ts = torch.LongTensor(ts)
        return hs, ts, in_data, out_data
        
        
        
    def text_subtensor(self, in_ctx, t_lens, out_ctx, h_lens): 
        in_ctx_data = [nn.utils.rnn.pad_sequence([torch.LongTensor(i) for i in in_ctx], 
                            batch_first=True, padding_value = PAD_TOKEN), torch.LongTensor(t_lens)]
        
        out_ctx_data = [nn.utils.rnn.pad_sequence([torch.LongTensor(i) for i in out_ctx], 
                            batch_first=True, padding_value = PAD_TOKEN), torch.LongTensor(h_lens)]
        
        if self.gpu:
            in_ctx_data = [x.cuda() for x in in_ctx_data]
            out_ctx_data = [x.cuda() for x in out_ctx_data]
        return in_ctx_data, out_ctx_data 

    
    
    def gen_subtensor(self, w_input, w_output, n_of_words): 
        w_input = nn.utils.rnn.pad_sequence([torch.LongTensor(i) for i in w_input], 
                                            batch_first=True, padding_value = PAD_TOKEN)
        w_output = nn.utils.rnn.pad_sequence([torch.LongTensor(i) for i in w_output], 
                                             batch_first=True, padding_value = PAD_TOKEN)
        input_lengths = torch.LongTensor(n_of_words)
        
        if self.gpu:
            w_input = w_input.cuda()
            w_output = w_output.cuda()
            input_lengths = input_lengths.cuda()
        return w_input, w_output, input_lengths
        
        

    def collate_fun(self, data):
        h_idx, t_idx, neg_h_idx, neg_t_idx, \
        h_ctxs, t_ctxs, h_ctx_len, t_ctx_len, \
        neg_h_ctxs, neg_t_ctxs, neg_h_ctx_len, neg_t_ctx_len, \
        w_input, w_output, n_of_words = zip(*data)

        neg_h_idx = [neg for negs in neg_h_idx for neg in negs]
        neg_t_idx = [neg for negs in neg_t_idx for neg in negs]   
        hs = list(h_idx)+ list(neg_h_idx)
        ts = list(t_idx) + list(neg_t_idx)

        h_ctxs = [neg_ctx for negs in h_ctxs for neg_ctx in negs]
        t_ctxs = [neg_ctx for negs in t_ctxs for neg_ctx in negs]
        neg_h_ctxs = [neg_ctx for negs in neg_h_ctxs for neg_ctx in negs]
        neg_t_ctxs = [neg_ctx for negs in neg_t_ctxs for neg_ctx in negs]

        h_ctx_len = [neg_ctx for negs in h_ctx_len for neg_ctx in negs]
        t_ctx_len = [neg_ctx for negs in t_ctx_len for neg_ctx in negs]
        neg_h_ctx_len = [neg_ctx for negs in neg_h_ctx_len for neg_ctx in negs]
        neg_t_ctx_len = [neg_ctx for negs in neg_t_ctx_len for neg_ctx in negs]

        h_ctxs = h_ctxs + neg_h_ctxs 
        t_ctxs = t_ctxs + neg_t_ctxs
        h_ctx_len = h_ctx_len + neg_h_ctx_len
        t_ctx_len = t_ctx_len + neg_t_ctx_len
    
        return hs, ts, h_ctxs, t_ctxs, h_ctx_len, t_ctx_len, w_input, w_output, n_of_words