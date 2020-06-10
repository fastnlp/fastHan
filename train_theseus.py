import os
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from fastNLP import (BucketSampler, Const, CrossEntropyLoss, DataSet,
                     DataSetIter, GradientClipCallback, LRScheduler,
                     SpanFPreRecMetric, Tester, Trainer, Vocabulary,
                     cache_results,RandomSampler,BucketSampler)
from fastNLP import GradientClipCallback, LRScheduler
from fastNLP.core.callback import WarmupCallback
#from model.multi_warmup import WarmupCallback
from fastNLP.core.optimizer import AdamW
from fastNLP.embeddings import BertEmbedding
from fastNLP.embeddings.static_embedding import StaticEmbedding
from fastNLP.io import DataBundle
from torch import nn as nn
from torch import optim as optim
from torch.optim.lr_scheduler import StepLR

from model.bert import BertEmbedding
from model.callbacks import DevCallback
from model.CharParser import CharParser
from model.metrics import CWSMetric, SegAppCharParseF1Metric
from model.model import CharModel
from model.AveMultiTaskIter import MultiTaskIter

lr = 3e-5   # 0.01~0.001
dropout = 0.1  # 0.3~0.6
arc_mlp_size = 500   # 200, 300
rnn_layers = 3  # 2, 3
label_mlp_size = 100

batch_size = 16
update_every = 1
n_epochs = 10
device = 3 if torch.cuda.is_available() else 'cpu'

target_list=['train','test','dev']
task_list=os.listdir('data/train')
all_data=torch.load('all_data')
chars_vocab=torch.load('chars_vocab')
label_vocab=torch.load('label_vocab')
pos_idx=chars_vocab.to_index('[unused14]')

model=torch.load('8l_model')

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



train_data=MultiTaskIter(all_data['train'],batch_size=batch_size,sampler=BucketSampler(batch_size=batch_size))

'''
for p in model.parameters():
    p.requires_grad=False
    '''
total_step=(len(train_data.dataset)//batch_size)*n_epochs*0.9
model.embed.model.encoder.encoder.init_theseus(total_step=total_step)
#model.embed.model.encoder.encoder.init_theseus(use_linear=False)

optimizer = AdamW(model.parameters(), lr=lr)

trainer = Trainer(train_data=train_data, model=model, optimizer=optimizer,
                  device=device,dev_data=all_data['test']['CWS-all'], batch_size=batch_size,
                  metrics=SpanFPreRecMetric(tag_vocab=label_vocab['CWS']),loss=None, n_epochs=n_epochs,check_code_level=-1,
                  update_every=update_every, test_use_tqdm=True,callbacks=callbacks)
trainer.train()

torch.save(model,'theseus_model')
