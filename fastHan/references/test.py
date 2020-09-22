import os

import torch
import torch.nn as nn
import torch.optim as optim
from fastNLP import (BucketSampler, Const, CrossEntropyLoss, DataSet,
                     DataSetIter, SpanFPreRecMetric, Tester, Trainer, Vocabulary,
                     cache_results,RandomSampler,BucketSampler)
from fastNLP.core.callback import WarmupCallback
from fastNLP.core.optimizer import AdamW
from torch import nn as nn

from model.bert import BertEmbedding
from model.metrics import CWSMetric, SegAppCharParseF1Metric
from model.model import CharModel
from model.AveMultiTaskIter import MultiTaskIter

lr = 2e-5   # 0.01~0.001
dropout = 0.1  # 0.3~0.6
arc_mlp_size = 500   # 200, 300
rnn_layers = 3  # 2, 3
label_mlp_size = 100

batch_size = 32
update_every = 1
n_epochs = 5

#prepare data
if False:
    label_vocab=dict()
    chars_vocab=torch.load('vocab/Parsing_chars')
    label_vocab['Parsing']=torch.load('vocab/Parsing_labels')
    target_list=['train','test','dev']
    task_list=os.listdir('data/train')
    all_data=dict()
    
    for target in target_list:
        all_data[target]=dict()
        for task in task_list:
            all_data[target][task]=torch.load('data/'+target+'/'+task)
    
    for task in all_data['train'].keys():
        if task=='Parsing-ctb9':
            continue
        dataset=all_data['train'][task]
        for word_lst in dataset['words']:
            chars_vocab.add_word_lst(word_lst)
            
    label_vocab['POS']=Vocabulary().from_dataset(all_data['train']['POS-ctb9'],field_name='target')
    label_vocab['CWS']=Vocabulary().from_dataset(all_data['train']['CWS-pku'],field_name='target')
    label_vocab['NER']=Vocabulary().from_dataset(all_data['train']['NER-msra'],field_name='target')
    label_vocab['Parsing']=torch.load('vocab/Parsing_labels')
    
    torch.save(all_data,'all_data')
    torch.save(chars_vocab,'chars_vocab')
    torch.save(label_vocab,'label_vocab')

else:
    target_list=['train','test','dev']
    task_list=os.listdir('data/train')
    all_data=torch.load('all_data')
    chars_vocab=torch.load('chars_vocab')
    label_vocab=torch.load('label_vocab')

for target in target_list:
    for task in task_list:
        if task=='Parsing-ctb9':
            all_data[target][task].rename_field('seq_lens','seq_len')
            all_data[target][task].drop(lambda  ins:ins['seq_len']>256)
            continue
        task_class=task.split('-')[0]
        chars_vocab.index_dataset(all_data[target][task], field_name='words',new_field_name='chars')
        label_vocab[task_class].index_dataset(all_data[target][task], field_name='target')
        all_data[target][task].set_input('chars','target','seq_len')
        all_data[target][task].set_target('target','seq_len')
        all_data[target][task].drop(lambda  ins:ins['seq_len']>256)

for target in ['test']:
    CWS_dataset=DataSet()
    for key in task_list:
        if key.startswith('CWS'):
            for ins in all_data[target][key]:
                CWS_dataset.append(ins)
            del all_data[target][key]
    CWS_dataset.set_input('chars','target','seq_len')
    CWS_dataset.set_target('target','seq_len')
    all_data[target]['CWS-all']=CWS_dataset

model=torch.load('best_model')


device = 0 if torch.cuda.is_available() else 'cpu'

metric1 = SegAppCharParseF1Metric(label_vocab['Parsing']['APP'])
metric2 = CWSMetric(label_vocab['Parsing']['APP'])
metrics = [metric1,metric2]



for key in all_data['test']:
    dataset=all_data['test'][key]
    if key.startswith('CWS'):
        tester = Tester(data=dataset,model=model,metrics=SpanFPreRecMetric(tag_vocab=label_vocab['CWS']),device=device)
    elif key.startswith('POS'):
        tester = Tester(data=dataset,model=model,metrics=SpanFPreRecMetric(tag_vocab=label_vocab['POS']),device=device)
    elif key.startswith('NER'):
        tester = Tester(data=dataset,model=model,metrics=SpanFPreRecMetric(tag_vocab=label_vocab['NER']),device=device)
    else:
        tester = Tester(data=dataset,model=model,metrics=metrics,device=device)
    print(key)
    tester.test()
