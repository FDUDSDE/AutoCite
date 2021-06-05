# -*- coding: utf-8 -*-
'''main script of the project.

This script contains a single function that execute the contextual citation generation task.
The function is diveded into the following phases:
 - Loading processed the data.
 - model graph construction
 - model training
 - model evaluation
'''

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn import metrics 
from rouge import Rouge 
from tqdm import tqdm

from parser import *
from utils import *
from models import Decoder
from datasets import HistDataset


def train(model, train_dl, epochs, optimizer):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.
        _step = 0
        with tqdm(total=len(train_dl), position=1, bar_format='{desc}') as desc:
            for batch in tqdm(train_dl, desc = '[Epoch {}]'.format(epoch+1), ncols = 80):
                hs, ts, out_ctx, in_ctx, h_lens, t_lens, w_input, w_output, n_of_words = [x for x in batch]
                
                # subgraph
                hs, ts, in_data, out_data = loader.graph_subtensor(hs, ts)
                in_ctx_data, out_ctx_data = loader.text_subtensor(in_ctx, t_lens, out_ctx, h_lens)
                w_input, w_output, input_lengths = loader.gen_subtensor(w_input, w_output, n_of_words) 

                link_loss, gen_loss, step_loss = model(hs, ts, 
                                                       in_data, out_data, 
                                                       in_ctx_data, out_ctx_data, 
                                                       w_input, w_output, input_lengths)
                
                # train
                model.zero_grad()
                step_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                epoch_loss += step_loss.cpu().detach()  
                del hs, ts, in_data, out_data, in_ctx_data, out_ctx_data, w_input, w_output, input_lengths
                
                _step += 1
                if _step % 10 == 1:
                    link_loss = round(link_loss.cpu().detach().item(), 2)
                    gen_loss = round(gen_loss.cpu().detach().item(), 2)
                    step_loss = round(step_loss.cpu().detach().item(), 2)
                    desc.set_description('[Train] #steps:{}\tgen:{}\tloss:{}'
                                         .format(link_loss, gen_loss, step_loss))
          
        desc.close()
        epoch_loss = round(epoch_loss.cpu().detach().item()/len(train_dl), 2)
        print('\nEpoch:{} \t loss:{} \n'.format(epoch, epoch_loss))
        return
  
    
def evaluation(model, test_dl, dataset):
    model.eval()
    total_auc = 0.
    rouge = Rouge()
    
    metrics_score = [[] for i in range(7)]
    generated_file = open('results/{}.txt'.format(dataset), "w") 
    
    _step = 0
    for batch in tqdm(test_dl, desc = '[evaluation]'):
        hs, ts, out_ctx, in_ctx, h_lens, t_lens, w_input, w_output, n_of_words = [x for x in batch]

        # sub-graph
        hs, ts, in_data, out_data = loader.graph_subtensor(hs, ts)
        in_ctx_data, out_ctx_data = loader.text_subtensor(in_ctx, t_lens, out_ctx, h_lens)
        link_head, link_tail, gen_head, gen_tail = model.evaluation_encoder(hs, ts, 
                                                                            in_data, out_data, 
                                                                            in_ctx_data, out_ctx_data)

        # link prediction
        head, tail = link_head, link_tail
        pos_num = len(w_input)
        neg_num = link_head.shape[0] - pos_num
        labels = torch.cat([torch.ones(pos_num), torch.zeros(neg_num)])
        scores = torch.sigmoid(torch.sum(torch.mul(head, tail), dim=1))
        auc = metrics.roc_auc_score(labels.cpu().detach().numpy(), scores.cpu().detach().numpy())
        total_auc += auc 

        # context generation
        head, tail = gen_head, gen_tail
        pos_head = head[:pos_num,:]
        pos_tail = tail[:pos_num,:]
        
        for i in range(pos_num):
            head = pos_head[i,:].unsqueeze(0)
            tail = pos_tail[i,:].unsqueeze(0)
            decoded_outputs = model.evaluate_decode(head, tail)
            source_sentence = ' '.join([loader.idx2word[idx] for idx in w_output[i]][:-1])
            generated_sentence = ' '.join([loader.idx2word[idx] for idx in decoded_outputs])
            
#             print('source_sentence:', source_sentence)
#             print('generated_sentence:', generated_sentence)
#             _step += 1
#             generated_file.write("Case :" + str(_step) + "\n")
#             generated_file.write('[source_sentence]\t' + source_sentence + "\n")
#             generated_file.write('[generated_sentence]\t' + generated_sentence + "\n\n")
            
            # BLEU
            bleu_1, bleu_2, bleu_3, bleu_4 = compute_bleu([source_sentence], [generated_sentence])
            metrics_score[0].append(bleu_1)          
            metrics_score[1].append(bleu_2)
            metrics_score[2].append(bleu_3)
            metrics_score[3].append(bleu_4)
            
            # ROUGE
            rouge_1 = rouge.get_scores(generated_sentence, source_sentence)
            metrics_score[4].append(rouge_1[0]['rouge-1']['f'])
            metrics_score[5].append(rouge_1[0]['rouge-2']['f'])
            metrics_score[6].append(rouge_1[0]['rouge-l']['f'])

        del hs, ts, in_data, out_data, in_ctx_data, out_ctx_data
    
    auc = total_auc/len(test_dl)    
    metrics_score = [sum(i)/len(i) for i in metrics_score]
    bleu_score = metrics_score[:4]
    rouge_score = metrics_score[4:7]
    return auc, bleu_score, rouge_score



if __name__ == '__main__':   
    #---------------- Loading data phase -----------------------
    opt = get_parser()
    loader = Loader(opt.dataset, opt.gpu)
  
    model = Decoder(n_words=loader.n_words,
                    n_nodes=loader.n_nodes,
                    max_len=loader.max_len,
                    opt = opt)
    if opt.gpu: model= model.cuda()
    
    if opt.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optim == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=opt.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
 

    #---------------------------- Training phase ----------------------------------
    print('Training...')
    train_dataset = HistDataset(loader, opt)
    train_dl = DataLoader(train_dataset, 
                          opt.batch_size, 
                          pin_memory=True, 
                          shuffle=True, 
                          collate_fn=loader.collate_fun, 
                          num_workers=1) 
    train(model, train_dl, opt.epochs, optimizer)
 

    #---------------------------- Evaluation phase ----------------------------
    print('Evaluation...')
    test_dataset = HistDataset(loader, opt, False)
    test_dl = DataLoader(test_dataset, 
                         opt.batch_size, 
                         pin_memory=True, 
                         collate_fn=loader.collate_fun) 
    auc, bleu_score, rouge_score = evaluation(model, test_dl, opt.dataset)
    
    print('AUC:{:.3f}'.format(auc))
    print('BLEU:', bleu_score)
    print('ROUGE:', rouge_score)
    
    #---------------------------------------------------------------------------
