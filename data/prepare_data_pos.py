from fastNLP import DataSet
import torch
from fastNLP.io import DataBundle
from fastNLP import Vocabulary
from fastNLP.io import CTBLoader
import re

bmes_data_path='cws_bmes/'
train_path='train_dataset/'
dev_path='dev_dataset/'
test_path='test_dataset/'

def normalize(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring


def process(instance):
    rNUM = '(-|\+)?\d+((\.|·)\d+)?%?'
    rENG = '[A-Za-z_.]+'
    new_sent = []
    for word in instance['raw_words']:
        word=normalize(word)
        word = re.sub('\s+', '', word, flags=re.U)
        word = re.sub(rNUM, '0', word, flags=re.U)
        word = re.sub(rENG, 'X', word)
        new_sent.append(word)
    return new_sent

def add_target(instance):
    pos=instance['pos']
    raw_words=instance['raw_words']
    target=[]
    for i in range(len(pos)):
        l=len(raw_words[i])
        if l==1:
            target.append('S-'+pos[i])
        else:
            target.append('B-'+pos[i])
            for j in range(l-2):
                target.append('M-'+pos[i])
            target.append('E-'+pos[i])
    return target

def add_raw_chars(instance):
    raw_chars=''.join(instance['raw_words'])
    return raw_chars

def add_chars(instance):
    chars=list(instance['raw_chars'])
    return chars

def process_dataset(dataset):
    dataset.delete_field('dep_head')
    dataset.delete_field('dep_label')
    dataset.apply(process,new_field_name='raw_words')
    dataset.apply(add_target,new_field_name='target')
    dataset.delete_field('pos')
    dataset.apply(add_raw_chars,new_field_name='raw_chars')
    dataset.apply(add_chars,new_field_name='chars')
    dataset.delete_field('raw_words')
    dataset.apply(lambda x:len(x['raw_chars']),new_field_name='seq_len')
    max_len=0
    for instance in dataset:
        if len(instance['target'])!=len(instance['chars']):
            print('error',instance)
        max_len=max(max_len,len(instance['chars']))
    print('max_len',max_len)
    return dataset

def load_ctb(path):
    data=CTBLoader().load(path)
    for key in data.datasets:
        data.datasets[key]=process_dataset(data.datasets[key])
        data.datasets[key].apply(lambda x:'POS-'+str(path)[:4],new_field_name='corpus')
    return data

#process ctb
path_list=['ctb5','ctb7','ctb9']
for path in path_list:
    print('start process'+path)
    databundle=load_ctb(path)
    torch.save(databundle.datasets['train'],train_path+'POS-'+path)
    torch.save(databundle.datasets['test'],train_path+'POS-'+path)
    torch.save(databundle.datasets['dev'],train_path+'POS-'+path)