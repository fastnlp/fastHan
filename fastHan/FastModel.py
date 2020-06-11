import os

import torch
from fastNLP import Vocabulary
from fastNLP.io.file_utils import cached_path

from .model.model import CharModel
from .model.bert import BertEmbedding


class Token(object):
    def __init__(self,word,pos=None,head=None,head_label=None,ner=None):
        self.word=word
        self.pos=pos
        self.head=head
        self.head_label=head_label
        self.ner=ner
    
    def __repr__(self):
        return self.word
    

class Sentence(object):
    def __init__(self,answer_list,target):
        self.answer_list=answer_list
        self.tokens=[]
        if target=='CWS':
            for word in answer_list:
                token=Token(word)
                self.tokens.append(token)
        elif target=='NER':
            for word,ner in answer_list:
                token=Token(word=word,ner=ner)
                self.tokens.append(token)
        elif target=='POS':
            for word,pos in answer_list:
                token=Token(word=word,pos=pos)
                self.tokens.append(token)
        else:
            for word,head,head_label,pos in answer_list:
                token=Token(word=word,pos=pos,head=head,head_label=head_label)
                self.tokens.append(token)
    
    def __repr__(self):
        return repr(self.answer_list)
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self,item):
        return self.tokens[item]


class FastHan(object):
    def __init__(self,model_type='base'):
        """
        #to be changed
        """
        self.device='cpu'
        #获取模型的目录/下载模型
        model_dir=self._get_model(model_type)

        #加载所需
        self.char_vocab=torch.load(os.path.join(model_dir,'chars_vocab'))
        self.label_vocab=torch.load(os.path.join(model_dir,'label_vocab'))
        model_path=os.path.join(model_dir,'model.bin')
        model_state_dict=torch.load(model_path)

        #创建新模型
        if model_type=='base':
            layer_num=4
        else:
            layer_num=8
        embed=BertEmbedding(self.char_vocab,model_dir_or_name=str(model_dir),dropout=0.1,include_cls_sep=False,layer_num=layer_num)
        self.model=CharModel(embed=embed,label_vocab=self.label_vocab)

        #复制参数
        self.model.load_state_dict(model_state_dict)

        self.model.to(self.device)
        self.model.eval()
        self.tag_map={'CWS':'[unused5]','POS':'[unused14]','NER':'[unused12]','Parsing':'[unused1]'}
        self.corpus_map={
        'CWS-as': '[unused2]',
        'CWS-cityu': '[unused3]',
        'CWS-cnc': '[unused4]',
        'CWS-ctb': '[unused5]',
        'CWS-msr': '[unused6]',
        'CWS-pku': '[unused7]',
        'CWS-sxu': '[unused8]',
        'CWS-udc': '[unused9]',
        'CWS-wtb': '[unused10]',
        'CWS-zx': '[unused11]',
        }

    def set_device(self,device):
        self.model.to(device)
        self.device=device
    
    def set_cws_style(self,corpus):
        corpus=corpus.lower()
        if not isinstance(corpus,str) or corpus not in ['as','cityu','cnc','ctb','msr','pku','sxu','udc','wtb','zx']:
            raise ValueError("corpus can only be string in ['as','cityu','cnc','ctb','msr',\
                'pku','sxu','udc','wtb','zx'].")
        corpus='CWS-'+corpus
        self.tag_map['CWS']=self.corpus_map[corpus]

    def _get_model(self,model_type):
        if model_type=='base':
            url='http://212.129.155.247/fasthan/fasthan_base.zip'
        elif model_type=='large':
            url='http://212.129.155.247/fasthan/fasthan_large.zip'
        else:
            raise ValueError("model_type can only be base or large.")
        
        model_dir=cached_path(url,name='fasthan')
        return model_dir
    
    
        
    def _to_tensor(self,chars,target,seq_len):
        task_class=[target]
        chars=torch.tensor(chars).to(self.device)
        seq_len=torch.tensor(seq_len).to(self.device)
        return chars,seq_len,task_class
        
    
    def _to_label(self,res,target):
        if target=='CWS':
            vocab=self.label_vocab['CWS']
        elif target=='POS':
            vocab=self.label_vocab['POS']
        elif target=='NER':
            vocab=self.label_vocab['NER']
        else:
            vocab=self.label_vocab['Parsing']
        
        ans=[]
        for label in res[1:]:
            ans.append(vocab.to_word(int(label)))
        return ans
        
    
    def _get_list(self,chars,tags):
        result=[]
        word=''
        for i in range(len(tags)):
            tag=tags[i]
            tag1,tag2=tag[0],tag[2:]
            if tag1=='B':
                word=''
                word+=chars[i]
            elif tag1=='S':
                if tag2=='':
                    result.append(chars[i])
                else:
                    result.append((chars[i],tag2))
            elif tag1=='E':
                word+=chars[i]
                if tag2=='':
                    result.append(word)
                else:
                    result.append([word,tag2])
            else:
                word+=chars[i]
        
        return result
    
    def _parsing(self,head_preds,label_preds,pos_label,chars):
        words=[]
        word=''
        for i in range(len(head_preds)):
            if label_preds[i]=='APP':
                word=word+chars[i]
            else:
                word=word+chars[i]
                words.append(word)
                word=''
        
        words=['root']+words
        lengths=[0]
        for i in range(1,len(words)):
            lengths.append(lengths[i-1]+len(words[i]))
        res=[]
        for i in range(1,len(lengths)):
            head=int(head_preds[lengths[i]-1])
            head=lengths.index(head)
            res.append([words[i],head,label_preds[lengths[i]-1],pos_label[lengths[i]-1].split('-')[1]])
        return res
    
    def __preprocess_sentence(self,sentence,target):
        tag=self.char_vocab.to_index(self.tag_map[target])
        max_len=max(map(len,sentence))
        chars=[]
        seq_len=[]
        for s in sentence:
            s=list(s.strip())
            char=[tag]
            for word in s:
                char.append(self.char_vocab.to_index(word))
            seq_len.append(len(char))
            char=char+[0]*(max_len-len(s))
            chars.append(char)
        chars,seq_len,task_class=self._to_tensor(chars,target,seq_len)
        return chars,seq_len,task_class
    
    def __call__(self,sentence,target='CWS'):
        if isinstance(sentence,str) and not isinstance(sentence,list):
             sentence=[sentence]
                
        elif isinstance(sentence,list):
            for s in sentence:
                if not isinstance(s,str):
                    raise ValueError("model can only parse string or list of string.")
        
        else:
            raise ValueError("model can only parse string or list of string.")
        
        if target not in ['CWS','NER','POS','Parsing']:
            raise ValueError("target can only be CWS, POS, NER or Parsing.")
        
        chars,seq_len,task_class=self.__preprocess_sentence(sentence,target)
        
        if target in ['CWS','POS','NER']:
            res=self.model.predict(chars,seq_len,task_class)['pred']
            ans=[]
            for i in range(chars.size()[0]):
                tags=self._to_label(res[i][:seq_len[i]],target)
                ans.append(self._get_list(sentence[i],tags))
        else:
            #parsing
            res=self.model.predict(chars,seq_len,task_class)
            ans=[]
            for i in range(chars.size()[0]):
                pos_label=self._to_label(res['pred'][i][:seq_len[i]],'POS')
                head_preds=res['head_preds'][i][1:seq_len[i]]
                label_preds=self._to_label(res['label_preds'][i][:seq_len[i]],'Parsing')
                ans.append(self._parsing(head_preds,label_preds,pos_label,sentence[i]))
        answer=[]
        for ans_ in ans:
            answer.append(Sentence(ans_,target))
        return answer