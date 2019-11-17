import torch
import torch.nn as nn
import torch.optim as optim
from fastNLP import (BucketSampler, Const, DataSet, DataSetIter,
                     GradientClipCallback, SpanFPreRecMetric, Trainer,Tester,
                     Vocabulary)
from fastNLP.embeddings import BertEmbedding
from fastNLP.io import DataBundle
from fastNLP.core.optimizer import AdamW

from new_model import Bert_Proj_CRF
from fastNLP.core.callback import WarmupCallback
import os


print(torch.cuda.current_device())
print(torch.cuda.device_count())


# load the prepared data
databundle=torch.load('databundle')
databundle.rename_field('chars','words')
def set_dataset(dataset):
    data=databundle.get_dataset(dataset)
    data.set_input('words','corpus','target')
    data.set_target('target','seq_len')
    data.set_pad_val('corpus',-1)
    data.set_pad_val('target',0)
set_dataset('train')
set_dataset('dev')
set_dataset('test')
print('have loaded the data')

lab_env=torch.cuda.is_available()
device = 2 if torch.cuda.is_available() else 'cpu'
callbacks = [WarmupCallback(warmup=0.1, schedule='linear'), ]

if False:
    model=torch.load('bestmodel')
    old_optimizer=torch.load('optimizer')
    optimizer=optim.Adam(model.parameters(),lr=1e-5,weight_decay=0.01)
    optimizer.load_state_dict(old_optimizer.state_dict())
    tester = Tester(data=databundle.get_dataset('test'),model=model,metrics=SpanFPreRecMetric(tag_vocab=databundle.vocabs[Const.TARGET], encoding_type='bmes'),device=device)

    tester.test()
else:
    if  False:
        embed=torch.load('embed')
    else:
        embed = BertEmbedding(databundle.get_vocab('words'), model_dir_or_name='cn-wwm-ext',dropout=0.1, include_cls_sep=False,layers='-1')
    model=Bert_Proj_CRF(embed=embed,tag_vocab=databundle.get_vocab('target'),dropout=0.1)
    optimizer = AdamW(model.parameters(), lr=2e-5)


trainer = Trainer(train_data=databundle.datasets['train'], model=model, optimizer=optimizer,
                  device=device,dev_data= databundle.datasets['dev'], batch_size=32,
                  metrics=SpanFPreRecMetric(tag_vocab=databundle.vocabs[Const.TARGET],encoding_type='bmes'),
                  loss=None, n_epochs=5,num_workers=2,sampler=BucketSampler(),
                  check_code_level=-1, update_every=4, test_use_tqdm=True,callbacks=callbacks)

print('train begin')
trainer.train()
torch.save(model,'bestmodel')
torch.save(optimizer,'optimizer')

tester = Tester(data=databundle.get_dataset('test'),model=model,metrics=SpanFPreRecMetric(tag_vocab=databundle.vocabs[Const.TARGET], encoding_type='bmes'),device=device)

tester.test()
