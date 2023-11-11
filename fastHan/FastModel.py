import os
import re
from shutil import copyfile

import torch
from fastNLP import Trainer
from fastNLP.core.optimizer import AdamW

from .model.bert import BertEmbedding
from .model.finetune_dataloader import (fastHan_CWS_Loader, fastHan_NER_Loader,
                                        fastHan_Parsing_Loader,
                                        fastHan_POS_loader)
from .model.model import CharModel
from .model.UserDict import UserDict
from .model.utils import hf_cached_path

class Token(object):
    """
    定义Token类。Token是fastHan输出的Sentence的组成单位，每个Token都代表一个被分好的词。
    它的属性则代表了它的词性、依存分析等信息。如果不包含此信息则为None。
    其中，word属性代表原词字符串；pos属性代表词性信息；head属性代表依存弧的指向（root为0，原句中的词，序号从1开始）；\
    head_label属性代表依存弧的标签；ner属性代表该词实体属性
    """

    def __init__(self,word,pos=None,head=None,head_label=None,ner=None,loc=None):
        self.word=word
        self.pos=pos
        #head代表依存弧的指向，root为0；原句中的词，序号从1开始。
        self.head=head
        #head_label代表依存弧的标签。
        self.head_label=head_label
        self.ner=ner
        self.loc=loc
    
    def __repr__(self):
        return self.word

    def __len__(self):
        return len(self.word)

    
    

class Sentence(object):
    """
    定义Sentence类。FastHan处理句子后会输出由Sentence组成的列表。
    Sentence由Token组成，可以用索引访问其中的Token。
    """


    def __init__(self,answer_list,target):
        self.answer_list=answer_list
        self.tokens=[]
        if target=='CWS':
            for word,loc in answer_list:
                token=Token(word,loc=loc)
                self.tokens.append(token)
        elif target=='NER':
            for word,ner,loc in answer_list:
                token=Token(word=word,ner=ner,loc=loc)
                self.tokens.append(token)
        elif target=='POS':
            for word,pos,loc in answer_list:
                token=Token(word=word,pos=pos,loc=loc)
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
    HF_URL_MAP = {
        "base": 'fdugzc/fasthan_base',
        "large": 'fdugzc/fasthan_large'
    }
    CACHE_SUB_DIR = "fasthan"

    def __init__(self,model_type='base',url=None):
        """
        初始化FastHan。包括获取模型参数目录，加载所需的词表、标签集，复制参数等。
        model_type共有base和large两种选择，分别是基于BERT前四层和前八层的模型。

        :param str model_type: 取值为'base'或'large'，决定模型的层数为4或8。

        :param str url:默认为None，用户可通过此参数传入手动下载并解压后的目录路径。
        """
        if model_type not in ["base","large"]:
            raise ValueError("model_type can only be base or large.")

        self.device='cpu'
        #获取模型的目录/下载模型
        if url is not None:
            model_dir=url
        else:
            model_dir=self._get_model(model_type)
        

        #加载所需词表、标签集、模型参数
        self.char_vocab=torch.load(os.path.join(model_dir,'chars_vocab'))
        self.label_vocab=torch.load(os.path.join(model_dir,'label_vocab'))
        self.model_dir=model_dir
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
        self.user_dict=UserDict()

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
        
    def finetune(self,data_path=None,task=None,lr=1e-5,n_epochs=1,batch_size=8,save=False,save_url=None):
        '''
        令fastHan继续在某项任务上进行微调。用户传入用于微调的数据集文件并设置超参数，fastHan自动加载并完成微调。此外，用户\
        可在微调后保存模型参数，并在初始化时通过url加载之前微调过的模型。进行微调的设备为当前fastHan的device。

        数据集文件的格式要求如下所示。
        对于CWS任务，则要求每行一条数据，每个词用空格分隔开。

        Example::

            上海 浦东 开发 与 法制 建设 同步
            新华社 上海 二月 十日 电 （ 记者 谢金虎 、 张持坚 ）
            ...

        对于NER任务，要求按照MSRA数据集的格式与标签集。
        Example::

            札 B-NS
            幌 E-NS
            雪 O
            国 O
            庙 O
            会 O
            。 O

            主 O
            道 O
            上 O
            的 O
            雪 O

            ...
        
        对于POS和dependency parsing，要求按照CTB9的格式与标签集。
        Example::

            1       印度    _       NR      NR      _       3       nn      _       _
            2       海军    _       NN      NN      _       3       nn      _       _
            3       参谋长  _       NN      NN      _       5       nsubjpass       _       _
            4       被      _       SB      SB      _       5       pass    _       _
            5       解职    _       VV      VV      _       0       root    _       _

            1       新华社  _       NR      NR      _       7       dep     _       _
            2       新德里  _       NR      NR      _       7       dep     _       _
            3       １２月  _       NT      NT      _       7       dep     _       _
            ...
        
        :param str data_path: 用于微调的数据集文件的路径
        :param str task: 此次微调的任务，可选值'CWS','POS','Parsing','NER'
        :param float lr: 微调的学习率。默认取1e-5
        :param int n_epochs: 微调的迭代次数，默认取1
        :param int batch_size: 每个batch的数据数量，默认为8。
        :param bool save: 是否保存微调后的模型，默认为False
        :param str save_url: 若保存模型，则此值为保存模型的路径
        '''
        assert(data_path and task)
        assert(task in ['CWS','POS','Parsing','NER'])
        if save:
            assert(save_url)
        if task=='CWS':
            dataset=fastHan_CWS_Loader(data_path, self.char_vocab, self.label_vocab['CWS'])
        elif task=='POS':
            dataset=fastHan_POS_loader(data_path,self.char_vocab,self.label_vocab['POS'])
        elif task=='NER':
            dataset=fastHan_NER_Loader(data_path,self.char_vocab,self.label_vocab['NER'])
        else:
            dataset=fastHan_Parsing_Loader(data_path,self.char_vocab,self.label_vocab)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

        trainer=Trainer(train_data=dataset, model=self.model, optimizer=optimizer,
                  device=self.device, batch_size=batch_size,
                  loss=None, n_epochs=n_epochs,check_code_level=-1,
                  update_every=1)
        trainer.train()
        self.model.eval()

        if save:
            if not os.path.exists(save_url):
                os.mkdir(save_url)
            print('saving model')
            copyfile(os.path.join(self.model_dir,'vocab.txt'),os.path.join(save_url,'vocab.txt'))
            copyfile(os.path.join(self.model_dir,'bert_config.json'),os.path.join(save_url,'bert_config.json'))
            copyfile(os.path.join(self.model_dir,'chars_vocab'),os.path.join(save_url,'chars_vocab'))
            copyfile(os.path.join(self.model_dir,'label_vocab'),os.path.join(save_url,'label_vocab'))
            torch.save(self.model.state_dict(),os.path.join(save_url,'model.bin'))

        return

    def add_user_dict(self,dic):
        '''
        为分词（CWS）任务添加用户词典。用户词典会在模型解码时根据词典改变标签序列的权重，令模型偏好根据用户词典进行解码。

        :param str,list dic:用户传入的词典，支持以下输入：
            1.str：词典文件的路径，词典文件中，每个词通过空格分隔。
            2.list：词典列表，列表中每个元素都是str形式的词。
        '''
        if isinstance(dic,str):
            self.user_dict.load_file(dic)
        elif isinstance(dic,list):
            self.user_dict.load_list(dic)
        else:
            raise ValueError("model can only parse string or list of string.")

    def remove_user_dict(self):
        '''
        移除当前的用户词典。
        '''
        self.user_dict = UserDict()

    def set_device(self,device):
        """
        调整模型至device。目前不支持多卡计算。

        :param str,torch.device,int device:支持以下的输入:
    
            1. str: ['cpu', 'cuda', 'cuda:0', 'cuda:1', ...] 依次为'cpu'中, 可见的第一个GPU中, 可见的第一个GPU中,
            可见的第二个GPU中;
    
            2. torch.device：将模型装载到torch.device上。
    
            3. int: 使用device_id为该值的gpu。
        """
        self.model.to(device)
        self.device=device
    
    def set_cws_style(self,corpus):
        """
        分词风格，指的是训练模型中文分词模块的10个语料库，模型可以区分这10个语料库，\
        设置分词style为S即令模型认为现在正在处理S语料库的分词。所以分词style实际上是\
        与语料库的覆盖面、分词粒度相关的。

        :param str corpus:语料库可选的取值为'as','cityu','cnc','ctb','msr','pku','sxu','udc','wtb','zx'。默认值为'ctb'。
        """
        corpus=corpus.lower()
        if not isinstance(corpus,str) or corpus not in ['as','cityu','cnc','ctb','msr','pku','sxu','udc','wtb','zx']:
            raise ValueError("corpus can only be string in ['as','cityu','cnc','ctb','msr',\
                'pku','sxu','udc','wtb','zx'].")
        corpus='CWS-'+corpus
        self.tag_map['CWS']=self.corpus_map[corpus]

    def _get_model(self, model_type):
        return hf_cached_path(FastHan.HF_URL_MAP[model_type], FastHan.CACHE_SUB_DIR)
            
    def _to_tensor(self,chars,target,seq_len):

        # 将向量序列、序列长度转换为pytorch中的tensor，从而输入CharModel。
        # 将当前任务变为列表，因为CharModel默认此参数为列表。

        task_class=[target]
        chars=torch.tensor(chars).to(self.device)
        seq_len=torch.tensor(seq_len).to(self.device)
        return chars,seq_len,task_class
           
    def _to_label(self,res,target):

        # 将一个向量序列转化为标签序列。

        # 根据当前任务选择标签集。

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
        
    def _get_list(self,chars,tags,return_loc=False):

        # 对使用BMES\BMESO标签集及交叉标签集的标签序列，输入原始字符串、标签集，转化为\
        # 分好词的Token序列，以及每个Token的属性。
        result=[]
        word=''
        for i in range(len(tags)):
            tag=tags[i]
            tag1,tag2=tag[0],tag[2:]
            if tag1=='B':
                word=''
                word+=chars[i]
                loc=i
            elif tag1=='S':
                if tag2=='':
                    if return_loc:
                        result.append([chars[i],i])
                    else:
                        result.append(chars[i])
                else:
                    if return_loc:
                        result.append([chars[i],tag2,i])
                    else:
                        result.append([chars[i],tag2])
                    
            elif tag1=='E':
                word+=chars[i]
                if tag2=='':
                    if return_loc:
                        result.append([word,loc])
                    else:
                        result.append(word)
                else:
                    if return_loc:
                        result.append([word,tag2,loc])
                    else:
                        result.append([word,tag2])
            else:
                word+=chars[i]
        
        return result
    
    def set_user_dict_weight(self,weight=0.05):
        '''
        设置词典分词结果对最终结果的影响程度。

        :param float weight:影响程度的权重，数值越大，分词结果越偏向于词典分词法。一般设为0.05-0.1即可有不错的结果。用户也可以自己构建dev set、test set来确定最适合自身任务的参数。
        '''

        assert(isinstance(weight,float) or isinstance(weight,int))
        self.model.user_dict_weight=weight
        return 0
       
    def _parsing(self,head_preds,label_preds,pos_label,chars):

        # 解析模型在依存分析的输出。

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
    
    #与用户词典相关。根据当前的任务，返回词典中B\M\E\S打头的tag索引
    def __get_tag_dict(self,task):
        important_tags=['B','M','E','S']
        result={}
        for tag in important_tags:
            result[tag]={}
        for tag in self.label_vocab[task].word2idx:
            if tag in important_tags:
                result[tag][tag]=self.label_vocab[task].word2idx[tag]
            elif '-' in tag and tag.split('-')[0] in important_tags:
                result[tag.split('-')[0]][tag]=self.label_vocab[task].word2idx[tag]
        return result
    
    def __preprocess_sentence(self,sentence,target):

        # 对原始的输入字符串、字符串列表做处理，转换为padding好的向量序列。

        tag=self.char_vocab.to_index(self.tag_map[target])
        max_len=max(map(len,sentence))
        chars=[]
        seq_len=[]
        for s in sentence:
            #对输入列表逐句处理
            s=list(s)
            char=[tag]
            for word in s:
                char.append(self.char_vocab.to_index(word))
            seq_len.append(len(char))
            char=char+[0]*(max_len-len(s))
            chars.append(char)
        #将已经数字化的输入序列转化为tensor
        chars,seq_len,task_class=self._to_tensor(chars,target,seq_len)
        return chars,seq_len,task_class
    
    def __call__(self,sentence,target='CWS',use_dict=False, return_list=True,return_loc=False):
        '''
        用户调用FastHan的接口函数。
        调用后会返回Sentence类。

        :param str,list sentence:用于解析的输入，可以是字符串形式的一个句子，也可以是由字符串形式的句子组成的列表。每个句子的长度需要小于等于512。

        :param str target:此次调用所进行的任务，可在'CWS','NER','POS','Parsing'中进行选择。其中'CWS','POS','Parsing'这几项任务的信息属于包含关系，调用更高层的任务可以返回互相兼容的多项任务的结果。

        :param bool use_dict:此次分词是否使用用户词典（如若使用，先调用add_user_dict），只有target为'CWS'时，才可令use_dict为True。

        :param bool return_list: 是否以list的形式将结果返回。默认为True，如果设为False，将以Sentence类的形式返回结果。
        '''
        #若输入的是字符串，转为一个元素的list
        if isinstance(sentence,str) and not isinstance(sentence,list):
             sentence=[sentence]
        
        elif isinstance(sentence,list):
            for s in sentence:
                if not isinstance(s,str):
                    raise ValueError("model can only parse string or list of string.")
        
        else:
            raise ValueError("model can only parse string or list of string.")

        if return_list is False and return_loc is False:
            return_loc=True
            print("return_list is False, return_loc is set to True")

        #去掉句子头尾空格
        for i,s in enumerate(sentence):
            sentence[i]=s.strip()
        
        if target not in ['CWS','NER','POS','Parsing']:
            raise ValueError("target can only be CWS, POS, NER or Parsing.")
        
        #对输入字符串列表做处理，转换为padding好的向量序列。
        if use_dict is True:
            if target=='Parsing':
                task='POS'
            else:
                task=target
            tag_seqs=[]
            for s in sentence:
                _,tag_seq=self.user_dict(s)
                tag_seq=['S']+tag_seq
                tag_seqs.append(tag_seq)
            max_len=max(map(len,tag_seqs))
            #t_代表tensor形式
            t_tag_seqs=torch.zeros([len(tag_seqs),max_len,len(self.label_vocab[task].word2idx)])
            important_tags=['B','M','E','S']
            tag_dict=self.__get_tag_dict(task)
            for i in range(len(tag_seqs)):
                for j in range(len(tag_seqs[i])):
                    if tag_seqs[i][j] in important_tags:
                        for tag in tag_dict[tag_seqs[i][j]].values():
                            t_tag_seqs[i][j][tag]=1
                        
                        #tag=self.label_vocab['CWS'].to_index(tag_seqs[i][j])
                        
            t_tag_seqs=t_tag_seqs.to(self.device)
            
        else:
            tag_seqs=None
            t_tag_seqs=None
                
        chars,seq_len,task_class=self.__preprocess_sentence(sentence,target)
        
        if target in ['CWS','POS','NER']:
            #如果当前任务是CWS、POS、NER，则进行如下处理

            #输入模型


            res=self.model.predict(chars,seq_len,task_class,t_tag_seqs)['pred']

            #将输出的一个batch的标签向量，逐句转换为分好的词以及每个词的属性。
            ans=[]
            for i in range(chars.size()[0]):
                tags=self._to_label(res[i][:seq_len[i]],target)
                ans.append(self._get_list(sentence[i],tags,return_loc=return_loc))
        else:
            #对依存分析进行如下处理

            #输入模型
            res=self.model.predict(chars,seq_len,task_class,t_tag_seqs)

            #逐句进行解析
            ans=[]
            for i in range(chars.size()[0]):
                #解析结果中的POS信息
                pos_label=self._to_label(res['pred'][i][:seq_len[i]],'POS')
                #解析依存分析信息
                head_preds=res['head_preds'][i][1:seq_len[i]]
                label_preds=self._to_label(res['label_preds'][i][:seq_len[i]],'Parsing')

                ans.append(self._parsing(head_preds,label_preds,pos_label,sentence[i]))
        
        if return_list:
            return ans

        #将结果转化为Sentence组成的列表
        answer=[]
        for ans_ in ans:
            answer.append(Sentence(ans_,target))
        return answer
