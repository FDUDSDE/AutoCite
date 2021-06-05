import numpy as np
import sys
import time
import random
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from beam import Beam

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(self.init_params(in_features, out_features), requires_grad=True)
        self.a1 = nn.Parameter(self.init_params(out_features, 1), requires_grad=True)
        self.a2 = nn.Parameter(self.init_params(out_features, 1), requires_grad=True)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def init_params(self, _in, _out):
        params = nn.init.xavier_uniform_(torch.FloatTensor(_in, _out), gain=np.sqrt(0.01))
        return params

    def forward(self, _input, adj):
        h = torch.mm(_input, self.W)
        N = h.size()[0]

        f_1 = h @ self.a1
        f_2 = h @ self.a2
        e = self.leakyrelu(f_1 + f_2.transpose(0, 1))  # node_num * node_num

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        # if self.concat:
        #   return F.elu(h_prime)
        # else:
        #   return h_prime
        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Encoder(nn.Module):
    ''' Graph Attention Embedding'''

    def __init__(self,
                 word_embedding,
                 node_embedding,
                 n_nodes,
                 n_features,
                 hidden_size,
                 dropout,
                 alpha,
                 nheads,
                 n_ctxs,
                 ctx_attn,
                 integration):
        super(Encoder, self).__init__()

        self.word_embedding = word_embedding
        self.node_embedding = node_embedding
        self.n_nodes = n_nodes
        self.out_feature = hidden_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_ctxs = n_ctxs
        self.ctx_attn = ctx_attn
        self.integration = integration

        # Graph Encoder
        self.outer_attentions = [GraphAttentionLayer(n_features,
                                                     self.out_feature // nheads,
                                                     dropout,
                                                     alpha,
                                                     True) for _ in range(nheads)]
        self.outer_out = nn.Linear(self.out_feature, self.out_feature)

        self.inner_attentions = [GraphAttentionLayer(n_features,
                                                     self.out_feature // nheads,
                                                     dropout,
                                                     alpha,
                                                     True) for _ in range(nheads)]
        self.inner_out = nn.Linear(self.out_feature, self.out_feature)

        # add modules
        for i, attention in enumerate(self.outer_attentions):
            self.add_module('outer_attention_{}'.format(i), attention)

        for i, attention in enumerate(self.inner_attentions):
            self.add_module('inner_attention_{}'.format(i), attention)

        # Text Encoder
        self.in_encoder_gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out_encoder_gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.in_s = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_s = nn.Linear(self.hidden_size, self.hidden_size)
        self.in_att = nn.Linear(self.hidden_size, 1)
        self.out_att = nn.Linear(self.hidden_size, 1)

        self.out_att_concat = nn.Linear(self.hidden_size * self.n_ctxs, self.hidden_size)
        self.in_att_concat = nn.Linear(self.hidden_size * self.n_ctxs, self.hidden_size)

        #  Gated Neural Fusion
        self.head_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.tail_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.link_multi_view_cat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gen_multi_view_cat = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.link_multi_view_add = nn.Linear(self.hidden_size, self.hidden_size)
        self.gen_multi_view_add = nn.Linear(self.hidden_size, self.hidden_size)

        self.multi_view_gate = nn.Embedding(self.n_nodes, self.hidden_size)  # 只用一个门控
        self.link_multi_view_gate = nn.Embedding(self.n_nodes, self.hidden_size)
        self.gen_multi_view_gate = nn.Embedding(self.n_nodes, self.hidden_size)

    def graph_encoder(self, in_data, out_data):
        in_nodes, inner_adj, in_map_hs, in_map_ts = in_data
        out_nodes, outer_adj, out_map_hs, out_map_ts = out_data

        out_nodes_emb = self.node_embedding(out_nodes)
        in_nodes_emb = self.node_embedding(in_nodes)

        # [n, emb_size]
        x = F.dropout(out_nodes_emb, self.dropout, training=self.training)
        out_emb = torch.cat([att(x, outer_adj) for att in self.outer_attentions], dim=1)

        # [n, hidden_size]
        x = F.dropout(in_nodes_emb, self.dropout, training=self.training)
        in_emb = torch.cat([att(x, inner_adj) for att in self.outer_attentions], dim=1)

        in_head = in_emb[in_map_hs]
        in_tail = in_emb[in_map_ts]
        out_head = out_emb[out_map_hs]
        out_tail = out_emb[out_map_ts]

        head = F.elu(in_head + out_head)
        tail = F.elu(in_tail + out_tail)
        head = F.dropout(head, self.dropout, training=self.training)
        tail = F.dropout(tail, self.dropout, training=self.training)
        return head, tail

    
    def gat_encoder(self, out_nodes, in_nodes, outer_adj, inner_adj):
        out_nodes_emb = self.node_embedding(out_nodes)
        in_nodes_emb = self.node_embedding(in_nodes)

        # [n, emb_size]
        x = F.dropout(out_nodes_emb, self.dropout, training=self.training)
        outer_x = torch.cat([att(x, outer_adj) for att in self.outer_attentions], dim=1)
        outer_x = F.dropout(outer_x, self.dropout, training=self.training)
        outer_x = self.outer_out(outer_x)

        # [n, hidden_size]
        x = F.dropout(in_nodes_emb, self.dropout, training=self.training)
        inner_x = torch.cat([att(x, inner_adj) for att in self.inner_attentions], dim=1)
        inner_x = F.dropout(inner_x, self.dropout, training=self.training)
        inner_x = self.inner_out(inner_x)
        return outer_x, inner_x

    
    def text_encoder(self, in_ctx_data, out_ctx_data):
        in_ctx, in_ctx_lengths = in_ctx_data
        out_ctx, out_ctx_lengths = out_ctx_data

        in_input_embedded = self.word_embedding(in_ctx)
        out_input_embedded = self.word_embedding(out_ctx)

        padded_in_input_embedded = nn.utils.rnn.pack_padded_sequence(in_input_embedded.transpose(1, 0),
                                                                     in_ctx_lengths, #.cpu()
                                                                     enforce_sorted=False)
        padded_out_input_embedded = nn.utils.rnn.pack_padded_sequence(out_input_embedded.transpose(1, 0),
                                                                      out_ctx_lengths, #.cpu() 
                                                                      enforce_sorted=False)

        in_output, in_hidden = self.in_encoder_gru(padded_in_input_embedded)
        out_output, out_hidden = self.out_encoder_gru(padded_out_input_embedded)

        if self.ctx_attn == 'concat':
            in_hidden = in_hidden.reshape([-1, self.hidden_size * self.n_ctxs])
            out_hidden = out_hidden.reshape([-1, self.hidden_size * self.n_ctxs])
            concat_in_hidden = self.in_att_concat(in_hidden)
            concat_out_hidden = self.out_att_concat(out_hidden)

            return concat_in_hidden, concat_out_hidden

        in_hidden = in_hidden.reshape([-1, self.n_ctxs, self.hidden_size])
        out_hidden = out_hidden.reshape([-1, self.n_ctxs, self.hidden_size])

        if self.ctx_attn == 'avg':
            avg_in_hidden = torch.mean(in_hidden, 1)
            avg_out_hidden = torch.mean(out_hidden, 1)
            return avg_out_hidden, avg_in_hidden

        elif self.ctx_attn == 'sum_co':
            attn_in2out = in_hidden.bmm(out_hidden.transpose(1, 2)).contiguous()
            attn_in2out = F.softmax(torch.sum(attn_in2out, 1), dim=1)

            attn_out2in = out_hidden.bmm(in_hidden.transpose(1, 2)).contiguous()
            attn_out2in = F.softmax(torch.sum(attn_out2in, 1), dim=1)

            out_hidden = attn_in2out.unsqueeze(1).bmm(out_hidden).squeeze()
            in_hidden = attn_out2in.unsqueeze(1).bmm(in_hidden).squeeze()
            return out_hidden, in_hidden

        elif self.ctx_attn == 'single':
            # single-attention
            in_hidden = in_hidden.reshape([-1, self.n_ctxs, self.hidden_size])  # .transpose(1,2)
            out_hidden = out_hidden.reshape([-1, self.n_ctxs, self.hidden_size])  # .transpose(1,2)

            in_attn = F.tanh(self.in_att(in_hidden))
            out_attn = F.tanh(self.out_att(out_hidden))

            in_attn = torch.softmax(in_attn, 1)
            out_attn = torch.softmax(out_attn, 1)

            attn_in_hidden = torch.bmm(in_attn.transpose(1, 2), in_hidden).squeeze()
            attn_out_hidden = torch.bmm(out_attn.transpose(1, 2), out_hidden).squeeze()
            return attn_out_hidden, attn_in_hidden

        '''
        in_output, _ = nn.utils.rnn.pad_packed_sequence(in_output)
        out_output, _ = nn.utils.rnn.pad_packed_sequence(out_output) 

        # co-attention
        in_output = in_output.transpose(0, 1)
        in_att = torch.unsqueeze(self.in_s(out_hidden.squeeze()), 2) 
        in_att_score = F.softmax(torch.bmm(in_output, in_att), dim=1)
        cross_in_hidden = torch.squeeze(torch.bmm(in_output.transpose(1,2), in_att_score))

        out_output = out_output.transpose(0, 1)
        out_att = torch.unsqueeze(self.out_s(in_hidden.squeeze()), 2) 
        out_att_score = F.softmax(torch.bmm(out_output, out_att), dim=1)
        cross_out_hidden = torch.squeeze(torch.bmm(out_output.transpose(1,2), out_att_score))

        return cross_in_hidden, cross_out_hidden
        '''
    

    def forward(self, hs, ts, in_data, out_data, in_ctx_data, out_ctx_data):
        # Graph-view
        graph_head, graph_tail = self.graph_encoder(in_data, out_data)
        # Text-view    
        text_head, text_tail = self.text_encoder(in_ctx_data, out_ctx_data)

        if self.integration == 'concat':
            head = torch.cat([graph_head, text_head], dim=1)
            tail = torch.cat([graph_tail, text_tail], dim=1)

            link_head = self.link_multi_view_cat(head)
            link_tail = self.link_multi_view_cat(tail)

            gen_head = self.gen_multi_view_cat(head)
            gen_tail = self.gen_multi_view_cat(tail)

            return link_head, link_tail, gen_head, gen_tail

        elif self.integration == 'add':
            head = graph_head + text_head
            tail = graph_tail + text_tail

            link_head = self.link_multi_view_add(head)
            link_tail = self.link_multi_view_add(tail)

            gen_head = self.gen_multi_view_add(head)
            gen_tail = self.gen_multi_view_add(tail)
            return link_head, link_tail, gen_head, gen_tail

        # head = torch.cat([graph_head, text_head], dim = 1)
        # tail = torch.cat([graph_tail, text_tail], dim = 1)

        # Gated-Combination 
        # gate_head = torch.sigmoid(self.multi_view_gate(hs))
        # gate_tail = torch.sigmoid(self.multi_view_gate(ts))

        # head = gate_head * graph_head + (1-gate_head) * text_head
        # tail = gate_tail * graph_tail + (1-gate_tail) * text_tail
        # return head, tail

        # Gated-Combination 
        link_gate_head = torch.sigmoid(self.link_multi_view_gate(hs))
        link_gate_tail = torch.sigmoid(self.link_multi_view_gate(ts))
        link_head = link_gate_head * graph_head + (1 - link_gate_head) * text_head
        link_tail = link_gate_tail * graph_tail + (1 - link_gate_tail) * text_tail

        gen_gate_head = torch.sigmoid(self.gen_multi_view_gate(hs))
        gen_gate_tail = torch.sigmoid(self.gen_multi_view_gate(ts))
        gen_head = gen_gate_head * graph_head + (1 - gen_gate_head) * text_head
        gen_tail = gen_gate_tail * graph_tail + (1 - gen_gate_tail) * text_tail

        return link_head, link_tail, gen_head, gen_tail


class Decoder(nn.Module):
    def __init__(self, n_words, n_nodes, max_len, opt):
        super(Decoder, self).__init__()
        self.n_words = n_words
        self.n_nodes = n_nodes
        self.max_len = max_len
        self.n_features = opt.n_features
        self.hidden_size = opt.hidden_size
        self.dropout = opt.dropout
        self.beam_size = opt.beam_size
        self.task = opt.task
        self.lamb = opt.lamb
        self.gpu = opt.gpu
        self.integration = opt.integration


        self.node_embedding = nn.Embedding(n_nodes, self.n_features)
        self.word_embedding = nn.Embedding(self.n_words, self.hidden_size)

        # Graph Encoder
        self.encoder = Encoder(word_embedding=self.word_embedding,
                               node_embedding=self.node_embedding,
                               n_nodes=self.n_nodes,
                               n_features=self.n_features,
                               hidden_size=self.hidden_size,
                               dropout=opt.dropout,
                               alpha=opt.alpha,
                               nheads=opt.nheads,
                               n_ctxs=opt.n_ctxs,
                               ctx_attn=opt.ctx_attn,
                               integration=self.integration)

        # Text Decoder
        self.head_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.tail_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.decode_gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.V = nn.Linear(self.hidden_size, self.n_words)

        # Loss 
        self.link_loss = nn.BCELoss()
        self.gen_loss = nn.CrossEntropyLoss()

    def init_params(self, _in, _out):
        params = nn.init.xavier_uniform_(torch.FloatTensor(_in, _out), gain=np.sqrt(2.0))
        return params

    def decode(self, head, tail, w_input, w_output, input_lengths):
        [batch_size, max_length] = w_input.shape

#         hidden = self.tanh(self.head_hidden(head) + self.tail_hidden(tail)).unsqueeze(0) 
#         input_embedded = self.word_embedding(w_input)
#         side_input = torch.cat([head, tail], dim = 1).unsqueeze(1).repeat(1, max_length, 1)
#         input_embedded = self.side_test(torch.cat((side_input, input_embedded), dim=2))

        input_embedded = self.word_embedding(w_input)
        hidden = self.tanh(self.head_hidden(head) + self.tail_hidden(tail)).unsqueeze(0)

        init_hidden = self.tanh(self.head_hidden(head) + self.tail_hidden(tail)).unsqueeze(1)
        # init_hidden = self.hidden_test(torch.cat([head, tail], dim = 1)).unsqueeze(1)
        init_word = F.softmax(self.V(init_hidden), dim=-1).topk(1)[1].squeeze()

        # init_word = self.V(init_hidden).topk(1)[1].squeeze()
        init_input_embedded = self.word_embedding(init_word).unsqueeze(1)
        input_embedded = torch.cat([init_input_embedded, input_embedded], dim=1)

        input_embedded = nn.utils.rnn.pack_padded_sequence(input_embedded,
                                                           input_lengths, #.cpu()
                                                           enforce_sorted=False,
                                                           batch_first=True)

        output, hidden = self.decode_gru(input_embedded, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

#         weighted_pvocab = self.V(output)
#         loss = self.gen_loss(weighted_pvocab.view(-1, self.n_words), w_output.view(-1))
#         return loss
        weighted_pvocab = F.softmax(self.V(output), dim=-1)
        target_id = w_output.unsqueeze(2)
        output = weighted_pvocab.gather(2, target_id).add_(sys.float_info.epsilon)
        target_mask_0 = target_id.ne(0).detach()
        loss = (output.log().mul(-1) * target_mask_0.float()).squeeze().sum(1).div(max_length)
        return loss.mean()
    

    def forward(self, hs, ts, in_data, out_data, in_ctx_data, out_ctx_data, w_input, w_output, input_lengths):

        link_head, link_tail, gen_head, gen_tail = self.encoder(hs,
                                                                ts,
                                                                in_data,
                                                                out_data,
                                                                in_ctx_data,
                                                                out_ctx_data)

        # link prediciton
        head, tail = link_head, link_tail
        pos_num = w_input.shape[0]
        neg_num = link_head.shape[0] - pos_num

        labels = torch.cat([torch.ones(pos_num), torch.zeros(neg_num)])
        scores = torch.sigmoid(torch.sum(torch.mul(head, tail), dim=1))
        if self.gpu: labels = labels.cuda()
        link_loss = self.link_loss(scores, labels)

        # context generation
        head, tail = gen_head, gen_tail
        pos_head = head[:pos_num, :]
        pos_tail = tail[:pos_num, :]
        gen_loss = self.decode(pos_head, pos_tail, w_input, w_output, input_lengths)

        if self.task == 'link':
            loss = self.lamb *link_loss
        elif self.task == 'gen':
            loss = gen_loss
        else:
            loss = self.lamb *link_loss + gen_loss
            # loss = link_loss + gen_loss 
        return link_loss, gen_loss, loss

    def evaluation_encoder(self, hs, ts, in_data, out_data, in_ctx_data, out_ctx_data):
        link_head, link_tail, gen_head, gen_tail = self.encoder(hs, ts,
                                                                in_data,
                                                                out_data,
                                                                in_ctx_data,
                                                                out_ctx_data)
        return link_head, link_tail, gen_head, gen_tail

    def evaluate_decode(self, head, tail):
        hidden = self.tanh(self.head_hidden(head) + self.tail_hidden(tail)).unsqueeze(0)
        output = hidden

        decoded_outputs = list()
        for _step in range(self.max_len):
            weighted_pvocab = F.softmax(self.V(output), dim=-1).squeeze()
            symbols = weighted_pvocab.topk(1)[1]
            if (symbols.detach().item() == EOS_TOKEN):
                break
            decoded_outputs.append(symbols.detach().item())
            w_input = symbols
            word_embedded = self.word_embedding(w_input).view(1, -1, self.hidden_size)
            # word_embedded = self.side_test(torch.cat((side_input, word_embedded), dim=2))  
            output, hidden = self.decode_gru(word_embedded, hidden)

        return decoded_outputs

    def evaluate_beam_decode(self, head, tail):
        hidden = self.tanh(self.head_hidden(head) + self.tail_hidden(tail)).unsqueeze(0)
        output = hidden

        # Initial beam search
        weighted_pvocab = F.softmax(self.V(output), dim=-1).squeeze()
        init_symbols = weighted_pvocab.topk(1)[1].repeat(self.beam_size)
        beam = Beam(self.beam_size, init_symbols, gpu=self.gpu)

        # Generation
        hidden = hidden.repeat(1, self.beam_size, 1)
        decoded_outputs = list()
        for _step in range(self.max_len):
            input = beam.get_current_state()
            word_embedded = self.word_embedding(input).view(-1, 1, self.hidden_size)
            output, hidden = self.decode_gru(word_embedded, hidden)
            word_lk = F.softmax(self.V(output.transpose(1, 0)), dim=-1)

            if beam.advance(word_lk.data): break

            hidden.data.copy_(hidden.data.index_select(1, beam.get_current_origin()))

        scores, ks = beam.sort_best()
        return beam.get_hyp(ks[0])
