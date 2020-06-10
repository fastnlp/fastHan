from fastNLP import DataSet
import torch
from fastNLP.io import DataBundle
from fastNLP import Vocabulary
from fastNLP.io import MsraNERLoader,OntoNotesNERLoader
import re

bmes_data_path='cws_bmes/'
train_path='train_dataset/'
dev_path='dev_dataset/'
test_path='test_dataset/'

msra=MsraNERLoader().load('MSRA')

def process(instance):
    rNUM = '(-|\+)?\d+((\.|·)\d+)?%?'
    rENG = '[A-Za-z_.]+'
    new_sent = []
    for word in instance['chars']:
        word=normalize(word)
        word = re.sub(rNUM, '0', word, flags=re.U)
        word = re.sub(rENG, 'X', word)
        new_sent.append(word)
    return new_sent

for key in msra.datasets:
    msra.datasets[key].rename_field('raw_chars','chars')
    msra.datasets[key].apply(process,new_field_name='chars')
    msra.datasets[key].apply(lambda ins:''.join(ins['chars']),new_field_name='raw_chars')
    msra.datasets[key].apply(lambda ins:len(ins['chars']),new_field_name='seq_len')
    msra.datasets[key].apply(lambda ins:'NER-msra',new_field_name='corpus')

train,dev=msra.datasets['train'].split(0.1)
torch.save(msra.datasets['test'],test_path+'NER-msra')
torch.save(train,train_path+'NER-msra')
torch.save(dev,dev_path+'NER-msra')

#OntoNotes

def process_word(word):
    rNUM = '(-|\+)?\d+((\.|·)\d+)?%?'
    rENG = '[A-Za-z_.]+'
    word=normalize(word)
    word = re.sub(rNUM, '0', word, flags=re.U)
    word = re.sub(rENG, 'X', word)
    return word

bmeso_data_path='Onto/'
def get_data_bmeso(dataset):
    path=bmeso_data_path+dataset+'.char.bmes'
    data={'raw_chars':[],'target':[],'seq_len':[],'corpus':[],'chars':[]}
    with open(path,encoding='UTF-8') as file:
        raw_sentence=[]
        tags=[]
        for line in file:
            if line=='\n' and len(raw_sentence)>0:
                data['raw_chars'].append(''.join(raw_sentence))
                data['target'].append(tags)
                data['seq_len'].append(len(tags))
                data['corpus'].append('NER-Onto')
                data['chars'].append(raw_sentence)
                raw_sentence=[]
                tags=[]
            else:
                word,tag=line.strip().split()
                word=process_word(word)
                raw_sentence.append(word)
                if tag.endswith('-PER'):
                    tag=tag[0]+'-NR'
                elif tag.endswith('-LOC'):
                    tag=tag[0]+'-NS'
                elif tag.endswith('-GPE'):
                    tag=tag[0]+'-NS'
                elif tag.endswith('-ORG'):
                    tag=tag[0]+'-NT'
                tags.append(tag)
        data=DataSet(data)
        return data

name='NER-Onto'
train_dataset=get_data_bmeso('train')
torch.save(train_dataset,train_path+name)
    
test_dataset=get_data_bmeso('test')
torch.save(test_dataset,test_path+name)

dev_dataset=get_data_bmeso('dev')
torch.save(dev_dataset,dev_path+name)