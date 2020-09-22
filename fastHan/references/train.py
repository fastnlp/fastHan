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
if True:
    label_vocab=dict()
    chars_vocab=Vocabulary(min_freq=2)
    target_list=['train','test','dev']
    task_list=os.listdir('data/train')
    all_data=dict()
    
    for target in target_list:
        all_data[target]=dict()
        for task in task_list:
            all_data[target][task]=torch.load('data/'+target+'/'+task)
    
    def change_tag(ins):
        words=['[unused14]']+ins['words'][1:]
        return words

    for target in target_list:
        all_data[target]['POS-ctb9'].apply(change_tag,new_field_name='words')


    print(all_data['train']['POS-ctb9'][0]['words'][:1])

    for task in all_data['train'].keys():
        if task.startswith('Parsing'):
            continue
        dataset=all_data['train'][task]
        for word_lst in dataset['words']:
            chars_vocab.add_word_lst(word_lst)

    pos_idx=chars_vocab.to_index('[unused14]')
    print(pos_idx)


    label_vocab['POS']=Vocabulary().from_dataset(all_data['train']['POS-ctb9'],field_name='target')
    label_vocab['CWS']=Vocabulary().from_dataset(all_data['train']['CWS-pku'],field_name='target')
    label_vocab['NER']=Vocabulary().from_dataset(all_data['train']['NER-msra'],field_name='target')
    label_vocab['Parsing']=torch.load('vocab/parsing_vocab')
    label_vocab['pos']=Vocabulary().from_dataset(all_data['train']['Parsing-ctb9'],field_name='pos')
        
    
    for target in target_list:
        for task in task_list:
            all_data[target][task].drop(lambda  ins:len(ins['words'])>256)
            chars_vocab.index_dataset(all_data[target][task], field_name='words',new_field_name='chars')
            task_class=task.split('-')[0]
            all_data[target][task].apply(lambda ins:task_class,new_field_name='task_class')
            if task=='Parsing-ctb9':
                label_vocab['Parsing'].index_dataset(all_data[target]['Parsing-ctb9'],field_name='char_labels')
                label_vocab[task_class].index_dataset(all_data[target][task], field_name='dep_label')
                label_vocab['pos'].index_dataset(all_data[target]['Parsing-ctb9'],field_name='pos')
                label_vocab['POS'].index_dataset(all_data[target]['Parsing-ctb9'],field_name='target')
                
                all_data[target][task].set_input('seq_len_for_wordlist','target','seq_len',
                                                 'chars','dep_head','dep_label','pos','word_lens','task_class')
                all_data[target][task].set_target('target','seq_len')
                all_data[target][task].set_target('seg_targets','seg_masks')
                all_data[target][task].set_target('gold_word_pairs','gold_label_word_pairs','pun_masks')
                
            else:
                label_vocab[task_class].index_dataset(all_data[target][task], field_name='target')
                
                all_data[target][task].set_input('chars','target','seq_len','task_class')
                all_data[target][task].set_target('target','seq_len')
                
            
    
    torch.save(all_data,'all_data')
    torch.save(chars_vocab,'chars_vocab')
    torch.save(label_vocab,'label_vocab')
            
            
else:
    target_list=['train','test','dev']
    task_list=os.listdir('data/train')
    all_data=torch.load('all_data')
    chars_vocab=torch.load('chars_vocab')
    label_vocab=torch.load('label_vocab')

embed=BertEmbedding(chars_vocab,model_dir_or_name='cn-wwm-ext',dropout=0.1,include_cls_sep=False,layer_num=8)

model=CharModel(embed=embed,
                label_vocab=label_vocab,
                pos_idx=pos_idx,
                Parsing_rnn_layers=rnn_layers,
                Parsing_arc_mlp_size=arc_mlp_size,
                Parsing_label_mlp_size=label_mlp_size,
                encoding_type='bmeso')


optimizer = AdamW(model.parameters(), lr=2e-5)

device = 0 if torch.cuda.is_available() else 'cpu'
callbacks = [WarmupCallback(warmup=0.1, schedule='linear') ]

metric1 = SegAppCharParseF1Metric(label_vocab['Parsing']['APP'])
metric2 = CWSMetric(label_vocab['Parsing']['APP'])
metric3 = SpanFPreRecMetric(tag_vocab=label_vocab['POS'])
metrics = [metric1,metric2,metric3]

for target in ['train','test','dev']:
    CWS_dataset=DataSet()
    for key in task_list:
        if key.startswith('CWS'):
            for ins in all_data[target][key]:
                CWS_dataset.append(ins)
            del all_data[target][key]
    CWS_dataset.set_input('chars','target','seq_len','task_class')
    CWS_dataset.set_target('target','seq_len')
    all_data[target]['CWS-all']=CWS_dataset


train_data=dict()
train_data['POS-ctb9']=all_data['train']['POS-ctb9']
train_data['CWS-all']=all_data['train']['CWS-all']
train_data=MultiTaskIter(all_data['train'],batch_size=batch_size,sampler=BucketSampler(batch_size=batch_size))

#del pos
trainer = Trainer(train_data=train_data, model=model, optimizer=optimizer,
                  device=device,dev_data=all_data['dev']['POS-ctb9'], batch_size=batch_size,
                  metrics=metric3,loss=None, n_epochs=n_epochs,check_code_level=-1,
                  update_every=update_every, test_use_tqdm=True,callbacks=callbacks)

trainer.train()
torch.save(model,'best_model')
