import os
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from fastNLP import (BucketSampler, Const, CrossEntropyLoss, DataSet,
                     DataSetIter, GradientClipCallback, LRScheduler,
                     SpanFPreRecMetric, Tester, Trainer, Vocabulary,
                     cache_results,RandomSampler)
from fastNLP.core.callback import WarmupCallback
from fastNLP.core.optimizer import AdamW
from fastNLP.embeddings import BertEmbedding
from fastNLP.embeddings.static_embedding import StaticEmbedding
from fastNLP.io import DataBundle
from torch import nn as nn
from torch import optim as optim
from torch.optim.lr_scheduler import StepLR

from model.metrics import CWSMetric, SegAppCharParseF1Metric
from model.model import CharModel
from model.MultiTaskIter import MultiTaskIter

#prepare data
target_list=['train','test','dev']
task_list=os.listdir('data/train')
all_data=torch.load('all_data')
chars_vocab=torch.load('chars_vocab')
label_vocab=torch.load('label_vocab')

print(len(label_vocab['POS']))
print(len(label_vocab['pos']))

for target in ['test']:
    CWS_dataset=DataSet()
    for key in task_list:
        if key.startswith('CWS'):
            for ins in all_data[target][key]:
                CWS_dataset.append(ins)
            del all_data[target][key]
    CWS_dataset.set_input('chars','target','seq_len','task_class')
    CWS_dataset.set_target('target','seq_len')
    all_data[target]['CWS-all']=CWS_dataset

model=torch.load('theseus_model')


device = 3 if torch.cuda.is_available() else 'cpu'

metric1 = SegAppCharParseF1Metric(label_vocab['Parsing']['APP'])
metric2 = CWSMetric(label_vocab['Parsing']['APP'])
metric3 = SpanFPreRecMetric(tag_vocab=label_vocab['POS'])
metrics = [metric1,metric2,metric3]



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
