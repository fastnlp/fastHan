from fastNLP import DataSet
import torch
from fastNLP.io import DataBundle
from fastNLP import Vocabulary

bmes_data_path='cws_bmes/'
train_path='train_dataset/'
dev_path='dev_dataset/'
test_path='test_dataset/'

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
                data['corpus'].append('CWS-'+raw_sentence[0][1:-1])
                data['chars'].append(raw_sentence[1:-1])
                raw_sentence=[]
                tags=[]
            else:
                word,tag=line.strip().split('\t')
                raw_sentence.append(word)
                tags.append(tag)
        data=DataSet(data)
        all_datasets={}
        vocab=Vocabulary(padding=None,unknown=None)
        vocab.from_dataset(data,field_name='corpus')
        for corpus in vocab:
            all_datasets[corpus[0]]=DataSet()
        for instance in data:
            all_datasets[instance['corpus']].append(instance)
        return all_datasets


train_datasets=get_data_bmes('train')
for key in train_datasets:
    dataset=train_datasets[key]
    torch.save(dataset,train_path+key)
    
dev_datasets=get_data_bmes('dev')
for key in dev_datasets:
    dataset=dev_datasets[key]
    torch.save(dataset,dev_path+key)
    
test_datasets=get_data_bmes('test')
for key in test_datasets:
    dataset=test_datasets[key]
    torch.save(dataset,test_path+key)