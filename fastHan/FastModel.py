import os

import torch
from fastNLP import Vocabulary
from fastNLP.io.file_utils import cached_path

from .model.model import CharModel
from .model.bert import BertEmbedding



class Token(object):
    """
    定义Token类。Token是fastHan输出的Sentence的组成单位，每个Token都代表一个被分好的词。
    它的pos等属性则代表了它的词性、依存分析等信息。如果不包含此信息则为None。
    """

    def __init__(self,word,pos=None,head=None,head_label=None,ner=None):
        self.word=word
        self.pos=pos
        #head代表依存弧的指向，root为0；原句中的词，序号从1开始。
        self.head=head
        #head_label代表依存弧的标签。
        self.head_label=head_label
        self.ner=ner
    
    def __repr__(self):
        return self.word
    

class Sentence(object):
    """
    定义Sentence类。FastHan处理句子后会输出由Sentence组成的列表。
    Sentence由Token组成，可以用索引访问其中的Token。
    """


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
    """
    FastHan类封装了基于BERT的深度学习联合模型CharModel，可处理CWS、POS、NER、dependency parsing四项任务，这\
    四项任务共享参数。
    """


    def __init__(self,model_type='base'):
        """
        初始化FastHan。包括获取模型参数目录，加载所需的词表、标签集，复制参数等。
        model_type共有base和large两种选择，分别是基于BERT前四层和前八层的模型。
        """
        self.device='cpu'
        #获取模型的目录/下载模型
        model_dir=self._get_model(model_type)

        #加载所需词表、标签集、模型参数
        self.char_vocab=torch.load(os.path.join(model_dir,'chars_vocab'))
        self.label_vocab=torch.load(os.path.join(model_dir,'label_vocab'))
        model_path=os.path.join(model_dir,'model.bin')
        model_state_dict=torch.load(model_path,map_location='cpu')

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

        #模型使用任务标签来区分任务、语料库，任务标签被映射到BERT词表中的[unused]
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
        """
        调整模型至device。如'cpu'，'cuda:0'。
        """
        self.model.to(device)
        self.device=device
    
    def set_cws_style(self,corpus):
        """
        分词风格，指的是训练模型中文分词模块的10个语料库，模型可以区分这10个语料库，\
        设置分词style为S即令模型认为现在正在处理S语料库的分词。所以分词style实际上是\
        与语料库的覆盖面、分词粒度相关的。
        corpus可在'as','cityu','cnc','ctb','msr','pku','sxu','udc','wtb','zx'中取值。
        """
        corpus=corpus.lower()
        if not isinstance(corpus,str) or corpus not in ['as','cityu','cnc','ctb','msr','pku','sxu','udc','wtb','zx']:
            raise ValueError("corpus can only be string in ['as','cityu','cnc','ctb','msr',\
                'pku','sxu','udc','wtb','zx'].")
        corpus='CWS-'+corpus
        self.tag_map['CWS']=self.corpus_map[corpus]

    def _get_model(self,model_type):
        """
        首先检查本地目录中是否已缓存模型，若没有缓存则下载。
        """
        if model_type=='base':
            url='http://212.129.155.247/fasthan/fasthan_base.zip'
        elif model_type=='large':
            url='http://212.129.155.247/fasthan/fasthan_large.zip'
        else:
            raise ValueError("model_type can only be base or large.")
        
        model_dir=cached_path(url,name='fasthan')
        return model_dir
    
    
        
    def _to_tensor(self,chars,target,seq_len):
        """
        将向量序列、序列长度转换为pytorch中的tensor，从而输入CharModel。
        将当前任务变为列表，因为CharModel默认此参数为列表。
        """
        task_class=[target]
        chars=torch.tensor(chars).to(self.device)
        seq_len=torch.tensor(seq_len).to(self.device)
        return chars,seq_len,task_class
        
    
    def _to_label(self,res,target):
        """
        将一个向量序列转化为标签序列。
        """
        #根据当前任务选择标签集。
        if target=='CWS':
            vocab=self.label_vocab['CWS']
        elif target=='POS':
            vocab=self.label_vocab['POS']
        elif target=='NER':
            vocab=self.label_vocab['NER']
        else:
            vocab=self.label_vocab['Parsing']
        #进行转换
        ans=[]
        for label in res[1:]:
            ans.append(vocab.to_word(int(label)))
        return ans
        
    
    def _get_list(self,chars,tags):
        """
        对使用BMES\BMESO标签集及交叉标签集的标签序列，输入原始字符串、标签集，转化为\
        分好词的Token序列，以及每个Token的属性。
        """
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
        """
        解析模型在依存分析的输出。
        """
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
        """
        对原始的输入字符串、字符串列表做处理，转换为padding好的向量序列。
        """
        tag=self.char_vocab.to_index(self.tag_map[target])
        max_len=max(map(len,sentence))
        chars=[]
        seq_len=[]
        for s in sentence:
            #对输入列表逐句处理
            s=list(s.strip())
            char=[tag]
            for word in s:
                char.append(self.char_vocab.to_index(word))
            seq_len.append(len(char))
            char=char+[0]*(max_len-len(s))
            chars.append(char)
        #将已经数字化的输入序列转化为tensor
        chars,seq_len,task_class=self._to_tensor(chars,target,seq_len)
        return chars,seq_len,task_class
    
    def __call__(self,sentence,target='CWS'):
        #若输入的是字符串，转为一个元素的list
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
        
        #对输入字符串列表做处理，转换为padding好的向量序列。
        chars,seq_len,task_class=self.__preprocess_sentence(sentence,target)
        
        if target in ['CWS','POS','NER']:
            #如果当前任务是CWS、POS、NER，则进行如下处理

            #输入模型
            res=self.model.predict(chars,seq_len,task_class)['pred']

            #将输出的一个batch的标签向量，逐句转换为分好的词以及每个词的属性。
            ans=[]
            for i in range(chars.size()[0]):
                tags=self._to_label(res[i][:seq_len[i]],target)
                ans.append(self._get_list(sentence[i],tags))
        else:
            #对依存分析进行如下处理

            #输入模型
            res=self.model.predict(chars,seq_len,task_class)

            #逐句进行解析
            ans=[]
            for i in range(chars.size()[0]):
                #解析结果中的POS信息
                pos_label=self._to_label(res['pred'][i][:seq_len[i]],'POS')
                #解析依存分析信息
                head_preds=res['head_preds'][i][1:seq_len[i]]
                label_preds=self._to_label(res['label_preds'][i][:seq_len[i]],'Parsing')

                ans.append(self._parsing(head_preds,label_preds,pos_label,sentence[i]))
        
        #将结果转化为Sentence组成的列表
        answer=[]
        for ans_ in ans:
            answer.append(Sentence(ans_,target))
        return answer