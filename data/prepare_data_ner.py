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