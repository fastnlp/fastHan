from fastNLP import DataSet
import torch
from fastNLP.io import DataBundle
from fastNLP import Vocabulary

bmes_data_path='data/bmes/'

def get_data_bmes(dataset):
    path=bmes_data_path+dataset+'.txt'
    data={'raw_chars':[],'target':[],'seq_len':[],'corpus':[],'chars':[]}
    with open(path,encoding='UTF-8') as file:
        raw_sentence=[]
        tags=[]
        for line in file:
            if line=='\n' and len(raw_sentence)>0:
                data['raw_chars'].append(''.join(raw_sentence[1:-1]))
                data['target'].append(tags[1:-1])
                data['seq_len'].append(len(tags)-2)
                data['corpus'].append(raw_sentence[0])
                data['chars'].append(raw_sentence[1:-1])
                raw_sentence=[]
                tags=[]
            else:
                word,tag=line.strip().split('\t')
                raw_sentence.append(word)
                tags.append(tag)
        return data

data=get_data_bmes('train')
train_dataset=DataSet(data)
data=get_data_bmes('dev')
dev_dataset=DataSet(data)
data=get_data_bmes('test')
test_dataset=DataSet(data)
datasets={'train':train_dataset,'dev':dev_dataset,'test':test_dataset}

vocab= Vocabulary()
vocab.from_dataset(train_dataset, field_name='chars')
for dataset in datasets.values():
    vocab.index_dataset(dataset, field_name='chars')
target_vocab = Vocabulary(padding=None, unknown=None)
target_vocab.from_dataset(train_dataset, field_name='target')
for dataset in datasets.values():
    target_vocab.index_dataset(dataset, field_name='target')
corpus_vocab = Vocabulary(padding=None, unknown=None)
corpus_vocab.from_dataset(train_dataset, field_name='corpus')
for dataset in datasets.values():
    corpus_vocab.index_dataset(dataset, field_name='corpus')    
vocabs={'chars':vocab,'target':target_vocab,'corpus':corpus_vocab}

databundle=DataBundle(vocabs=vocabs,datasets=datasets)
print(databundle)
torch.save(databundle,'databundle')