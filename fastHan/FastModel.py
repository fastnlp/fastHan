import os
import torch
import json
import pickle

from shutil import copyfile
from transformers import AdamW, AutoModel
from fastNLP.io.file_utils import cached_path
from transformers import AutoTokenizer
from fastNLP import DataSet
from fastNLP.core import LoadBestModelCallback, TorchWarmupCallback
from fastNLP import Trainer, Vocabulary, BucketedBatchSampler
from fastNLP.core.dataloaders import TorchDataLoader
from .model.baseModel import MultiTaskModel
from .model.UserDict import UserDict
from .model.multitask_metric_base import MultiTaskMetric
from .model.finetune_dataloader import fastHan_CWS_Loader, fastHan_NER_Loader, fastHan_Parsing_Loader, fastHan_POS_guwen_loader, fastHan_POS_loader
from .model.weight_manager import FixedWeightManager, EnsembledWeightManagers, convert_cws_macro

MAX_LEN = 300
from transformers.utils import logging

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Token(object):
    """
    定义Token类。Token是fastHan输出的Sentence的组成单位, 每个Token都代表一个被分好的词。
    它的属性则代表了它的词性、依存分析等信息。如果不包含此信息则为None。
    其中:
        word属性代表原词字符串
        pos属性代表词性信息
        head属性代表依存弧的指向, root为0, 原句中的词的序号从1开始
        head_label属性代表依存弧的标签
        ner属性代表该词实体属性
        loc是否以Sentence的形式返回
    """
    def __init__(self,
                 word,
                 pos=None,
                 head=None,
                 head_label=None,
                 ner=None,
                 loc=None):
        self.word = word
        self.pos = pos
        #head代表依存弧的指向，root为0；原句中的词，序号从1开始。
        self.head = head
        #head_label代表依存弧的标签。
        self.head_label = head_label
        self.ner = ner
        self.loc = loc

    def __repr__(self):
        return self.word

    def __len__(self):
        return len(self.word)


class Sentence(object):
    """
    定义Sentence类。FastHan处理句子后会输出由Sentence组成的列表。
    Sentence由Token组成, 可以用索引访问其中的Token。
    """
    def __init__(self, answer_list, target):
        self.answer_list = answer_list
        self.tokens = []
        if target == 'CWS':
            for word, loc in answer_list:
                token = Token(word, loc=loc)
                self.tokens.append(token)
        elif target == 'POS':
            for word, pos, loc in answer_list:
                token = Token(word=word, pos=pos, loc=loc)
                self.tokens.append(token)
        elif target == 'NER':
            for word, ner, loc in answer_list:
                token = Token(word=word, ner=ner, loc=loc)
                self.tokens.append(token)
        else:
            for word, head, head_label in answer_list:
                token = Token(word=word, head=head, head_label=head_label)
                self.tokens.append(token)

    def __repr__(self):
        return repr(self.answer_list)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        return self.tokens[item]


class FastHan(object):
    """
        FastHan可处理:
            CWS、POS、CWS-guwen、POS-guwen、NER、dependency parsing任务, 
            这几项任务共享参数。
    """
    def __init__(self, model_type='base', url=None):
        self.device = 'cpu'
        #获取模型的目录/下载模型
        if url is not None:
            model_dir = url
        else:
            model_dir = self._get_model(model_type)
        # 加载预训练时所有的任务名, 每种任务代表了不同的CWS、POS等的规范
        # 任务名形如: CWS-pku
        with open(os.path.join(model_dir, 'all_tasks.json'),
                  'r',
                  encoding='utf-8') as f:
            self.all_tasks = json.load(f)
        self.task_to_index = dict([(self.all_tasks[i], i)
                                   for i in range(len(self.all_tasks))])
        # 加载各种规范下的标签集合
        with open(os.path.join(model_dir, 'task_label_map.json'),
                  'r',
                  encoding='utf-8') as f:
            task_label_map = json.load(f)
        task_vocab_map = dict()
        for task in self.all_tasks:
            if task.startswith('CWS'):
                vocab = Vocabulary(padding=None, unknown=None)
                vocab.add_word_lst(['b', 'e', 'm', 's'])
            else:
                vocab = Vocabulary(padding=None, unknown=None)
                vocab.add_word_lst(task_label_map[task])
            task_vocab_map[task] = vocab
        self.task_vocab_map = task_vocab_map

        # weight_manager
        managers = []
        with open(os.path.join(model_dir, 'weight_v0.bin'), 'rb') as f:
            weight = pickle.load(f)
        manager = FixedWeightManager(all_tasks=self.all_tasks,
                                     weight=weight,
                                     key_mapper=convert_cws_macro(
                                         self.all_tasks))
        managers.append([manager, -1])
        with open(os.path.join(model_dir, 'corpus_base.bin'), 'rb') as f:
            weight = pickle.load(f)
        manager = FixedWeightManager(all_tasks=self.all_tasks, weight=weight)
        managers.append([manager, 1])
        self.ensembledWeightManager = EnsembledWeightManagers(managers)

        self.model_dir = model_dir
        # 加载模型
        model_path = os.path.join(model_dir, 'fasthan-model.pth')
        self.base_model_name = 'hfl/chinese-bert-wwm'
        if model_type == 'base':
            layer_num = 12
        else:
            layer_num = 12
        encoder = AutoModel.from_pretrained(self.base_model_name,
                                            num_hidden_layers=layer_num)
        self.model = MultiTaskModel(
            encoder,
            task_label_map,
            self.all_tasks,
            ensembledWeightManager=self.ensembledWeightManager)
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.to(self.device)
        self.model.eval()
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        # 设置默认的任务风格匹配
        self.style = {
            "CWS": "CWS-cnc",
            "POS": "POS-ctb9",
            "CWS-guwen": "POS-guwen",
            "POS-guwen": "POS-guwen",
            "NER": "NER-msra",
            "Parsing": "Parsing-ctb9"
        }
        # 设置用户自定义字典
        self.user_dict = UserDict()
        self.user_dict_weight = 0.05

    def finetune(self,
                 data_path=None,
                 task=None,
                 lr=1e-5,
                 n_epochs=1,
                 batch_size=8,
                 save=False,
                 save_url=None):
        '''
        令fastHan继续在某项任务上进行微调。
            1. 用户传入用于微调的数据集文件并设置超参数, fastHan自动加载并完成微调。
            2. 用户可在微调后保存模型参数,并在初始化时通过url加载之前微调过的模型。
            3. 进行微调的设备为当前fastHan的device。

        数据集文件的格式要求如下所示。
        -- 对于CWS任务, 则要求每行一条数据, 每个词用空格分隔开。
        Example::
            上海 浦东 开发 与 法制 建设 同步
            新华社 上海 二月 十日 电 （ 记者 谢金虎 、 张持坚 ）
            ...

        -- 对于NER任务, 要求按照MSRA数据集的格式与标签集。
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
        
        -- 对于POS和dependency parsing, 要求按照CTB9的格式与标签集。
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
        
        -- 对于CWS-guwen, 由于训练样本制约, 本模型暂不支持单纯的对古文分词进行训练。

        -- 对于POS-guwen, 要求按照如下格式, 每个句子一行。
        Example::
            春秋/n 左傳/n 定公/nr
            元年/t ，/w 春/n ，/w 王/n 正月/t 辛巳/t ，/w 晉/ns 魏舒/nr 合/v 諸侯/n 之/u 大夫/n 于/p 狄泉/ns ，/w 將/d 以/c 城/n 成周/ns 。/w
            魏子/nr 蒞政/v 。/w

        :param str data_path: 用于微调的数据集文件的路径。
        :param str task: 此次微调的任务，可选值'CWS','POS','CWS-guwen','POS-guwen','Parsing','NER'。
        :param float lr: 微调的学习率。默认取1e-5。
        :param int n_epochs: 微调的迭代次数, 默认取1。
        :param int batch_size: 每个batch的数据数量, 默认为8。
        :param bool save: 是否保存微调后的模, 默认为False。
        :param str save_url: 若保存模型，则此值为保存模型的路径。
        '''
        assert (data_path and task)
        assert (task
                in ['CWS', 'POS', 'CWS-guwen', 'POS-guwen', 'Parsing', 'NER'])
        with open(data_path, encoding='utf-8') as f:
            lines = f.readlines()
        if save:
            assert (save_url)
        if task == 'CWS':
            dataset = fastHan_CWS_Loader(lines,
                                         self.task_vocab_map[self.style[task]],
                                         self.tokenizer)
        elif task == 'POS':
            dataset = fastHan_POS_loader(lines,
                                         self.task_vocab_map[self.style[task]],
                                         self.tokenizer)
        elif task == 'CWS-guwen':
            print(
                "[Warning]: You can not finetune CWS-guwen only, you can finetune on POS-guwen dataset."
            )
            return
        elif task == 'POS-guwen':
            dataset = fastHan_POS_guwen_loader(
                lines, self.task_vocab_map[self.style[task]], self.tokenizer)
        elif task == 'NER':
            dataset = fastHan_NER_Loader(lines,
                                         self.task_vocab_map[self.style[task]],
                                         self.tokenizer)
        else:
            dataset = fastHan_Parsing_Loader(
                lines, self.task_vocab_map[self.style[task]], self.tokenizer)

        dataset.set_pad('labels', -100)
        if task == 'Parsing':
            dataset.set_pad('heads', -100)
        dataset.add_field('task', [self.task_to_index[self.style[task]]] *
                          len(dataset))
        dataloader = TorchDataLoader(dataset,
                                     batch_sampler=BucketedBatchSampler(
                                         dataset=dataset,
                                         batch_size=batch_size,
                                         length='seq_len'))
        metrics = {'f': MultiTaskMetric(self.all_tasks, self.task_vocab_map)}
        optimizer = AdamW(self.model.parameters(), lr=lr)
        monitor = 'avg_f#f'
        load_callback = LoadBestModelCallback(monitor=monitor,
                                              delete_after_train=False)
        warmup_callback = TorchWarmupCallback()
        callbacks = [load_callback, warmup_callback]

        trainer = Trainer(train_dataloader=dataloader,
                          model=self.model,
                          optimizers=optimizer,
                          device=self.device,
                          metrics=metrics,
                          evaluate_dataloaders=dataloader,
                          n_epochs=n_epochs,
                          progress_bar='tqdm',
                          callbacks=callbacks)
        self.model.train()
        trainer.run()
        self.model.eval()

        if save:
            if not os.path.exists(save_url):
                os.mkdir(save_url)
            print('saving model')
            copyfile(os.path.join(self.model_dir, 'all_tasks.json'),
                     os.path.join(save_url, 'all_tasks.json'))
            copyfile(os.path.join(self.model_dir, 'task_label_map.json'),
                     os.path.join(save_url, 'task_label_map.json'))
            copyfile(os.path.join(self.model_dir, 'weight_v0.bin'),
                     os.path.join(save_url, 'weight_v0.bin'))
            copyfile(os.path.join(self.model_dir, 'corpus_base.bin'),
                     os.path.join(save_url, 'corpus_base.bin'))
            torch.save(self.model.state_dict(),
                       os.path.join(save_url, 'fasthan-model.pth'))
        return

    def add_user_dict(self, dic):
        '''
        添加用户词典。
            用户词典会在模型解码时根据词典改变标签序列的权重，令模型偏好根据用户词典进行解码。
        :param str,list dic: 用户传入的词典，支持以下输入：
            1.str: 词典文件的路径, 词典文件中, 每个词通过空格分隔。
            2.list: 词典列表, 列表中每个元素都是str形式的词。
        '''
        if isinstance(dic, str):
            self.user_dict.load_file(dic)
        elif isinstance(dic, list):
            self.user_dict.load_list(dic)
        else:
            raise ValueError("model can only parse string or list of string.")

    def remove_user_dict(self):
        '''
        移除当前的用户词典。
        '''
        self.user_dict = UserDict()

    def set_user_dict_weight(self, weight=0.05):
        '''
        设置词典分词结果对最终结果的影响程度。
        :param float weight:
            影响程度的权重, 数值越大, 分词结果越偏向于词典分词法。
            一般设为0.05-0.1即可有不错的结果。
            用户也可以自己构建dev set、test set来确定最适合自身任务的参数。
        '''
        assert (isinstance(weight, float) or isinstance(weight, int))
        self.user_dict_weight = weight

    def _get_model(self, model_type):
        """
            首先检查本地目录中是否已缓存模型，若没有缓存则下载。
        """
        if model_type == 'base':
            url = 'http://download.fastnlp.top/fasthan/fasthan.zip'
        elif model_type == 'large':
            url = 'http://download.fastnlp.top/fasthan/fasthan.zip'
        else:
            raise ValueError("model_type can only be base or large.")

        model_dir = cached_path(url, name='fasthan')
        return model_dir

    def set_device(self, device):
        """
        调整模型至device。目前不支持多卡计算。
        :param str,torch.device,int device:支持以下的输入:
            1. str: ['cpu', 'cuda', 'cuda:0', 'cuda:1', ...] 依次为'cpu'中, 可见的第一个GPU中, 可见的第一个GPU中,
            可见的第二个GPU中;
            2. torch.device: 将模型装载到torch.device上。
            3. int: 使用device_id为该值的gpu。
        """
        self.model.to(device)
        self.device = device

    def set_cws_style(self, corpus):
        """
        分词风格，指的是训练模型中文分词模块的10个语料库，模型可以区分这10个语料库，\
        设置分词style为S即令模型认为现在正在处理S语料库的分词。所以分词style实际上是\
        与语料库的覆盖面、分词粒度相关的。

        :param str corpus:语料库可选的取值为'as','cityu','cnc','msr','pku','sxu','udc','wtb','zx'。默认值为'ctb'。
        """
        corpus = corpus.lower()
        if not isinstance(corpus, str) or corpus not in [
                'as', 'cityu', 'cnc', 'msr', 'pku', 'sxu', 'udc', 'wtb', 'zx'
        ]:
            raise ValueError(
                "corpus can only be string in ['as','cityu','cnc','msr',\
                'pku','sxu','udc','wtb','zx'].")
        corpus = 'CWS-' + corpus
        self.style['CWS'] = corpus

    def _to_label(self, res, task_class):
        # 将一个向量序列转化为标签序列。
        # 根据当前任务选择标签集。
        vocab = self.task_vocab_map[task_class]
        #进行转换
        ans = []
        for label in res:
            ans.append(vocab.to_word(int(label)))
        return ans

    def _get_list(self, chars, tags, target, return_loc=False):
        # 对使用BMES\BMESO标签集及交叉标签集的标签序列，输入原始字符串、标签集，转化为\
        # 分好词的Token序列，以及每个Token的属性。
        result = []
        word = ''
        for i in range(len(tags)):
            tag = tags[i]
            tag1, tag2 = tag[0], tag[2:]
            if tag1 == 'b':
                word = ''
                word += chars[i]
                loc = i
            elif tag1 == 's':
                if tag2 == '' or target == 'CWS-guwen':
                    if return_loc:
                        result.append([chars[i], i])
                    else:
                        result.append(chars[i])
                else:
                    if return_loc:
                        result.append([chars[i], tag2, i])
                    else:
                        result.append([chars[i], tag2])

            elif tag1 == 'e':
                word += chars[i]
                if tag2 == '' or target == 'CWS-guwen':
                    if return_loc:
                        result.append([word, loc])
                    else:
                        result.append(word)
                else:
                    if return_loc:
                        result.append([word, tag2, loc])
                    else:
                        result.append([word, tag2])
            else:
                word += chars[i]
        return result

    def _parsing(self, head_preds, label_preds, chars):
        # 解析模型在依存分析的输出。
        words = []
        word = ''
        for i in range(len(head_preds)):
            if label_preds[i] == 'app':
                word = word + chars[i]
            else:
                word = word + chars[i]
                words.append(word)
                word = ''

        words = ['root'] + words
        lengths = [0]
        for i in range(1, len(words)):
            lengths.append(lengths[i - 1] + len(words[i]))
        res = []
        for i in range(1, len(lengths)):
            head = int(head_preds[lengths[i] - 1])
            head = lengths.index(head)
            res.append([words[i], head, label_preds[lengths[i] - 1]])
        return res

    #与用户词典相关。根据当前的任务，返回词典中b, e, m, s打头的tag索引
    def __get_tag_dict(self, task):
        important_tags = ['b', 'e', 'm', 's']
        result = {}
        for tag in important_tags:
            result[tag] = {}
        for tag in self.task_vocab_map[task].word2idx:
            if tag in important_tags:
                result[tag][tag] = self.task_vocab_map[task].word2idx[tag]
            elif '-' in tag and tag.split('-')[0] in important_tags:
                result[tag.split('-')
                       [0]][tag] = self.task_vocab_map[task].word2idx[tag]
        return result

    def __preprocess_sentence(self, sentence, tag_seqs, target):
        data = {'input_ids': [], 'attention_mask': [], 'seq_len': []}

        for idx, line in enumerate(sentence):
            line = line.strip()
            line = line[:MAX_LEN]
            tokenize_result = self.tokenizer([line], is_split_into_words=True)
            if target == 'Parsing':
                # 添加根结点
                tokenize_result['input_ids'].insert(1, 1)
                tokenize_result['attention_mask'].insert(1, 1)
            data['input_ids'].append(tokenize_result['input_ids'])
            data['attention_mask'].append(tokenize_result['attention_mask'])
            data['seq_len'].append(len(tokenize_result['input_ids']) - 2)
        if tag_seqs is not None:
            data['tag_seqs'] = tag_seqs
        dataset = DataSet(data)
        dataset.set_pad('labels', -100)
        if target == 'Parsing':
            dataset.set_pad('heads', -100)

        task_class = self.style[target]
        dataset.add_field('task',
                          [self.task_to_index[task_class]] * len(dataset))
        dataloader = TorchDataLoader(dataset,
                                     batch_sampler=BucketedBatchSampler(
                                         dataset=dataset,
                                         batch_size=1,
                                         length='seq_len'))
        return dataloader, task_class

    def __call__(self,
                 sentence,
                 target='CWS',
                 use_dict=False,
                 return_list=True,
                 return_loc=False):
        '''
            用户调用FastHan的接口函数。
            调用后会返回Sentence类。

            :param str,list sentence:
                用于解析的输入, 可以是字符串形式的一个句子, 也可以是由字符串形式的句子组成的列表。
                每个句子的长度需要小于等于300。
            :param str target:
                此次调用所进行的任务，可在'CWS','NER','POS','Parsing'中进行选择。
                其中'CWS','POS','Parsing'这几项任务的信息属于包含关系,
                调用更高层的任务可以返回互相兼容的多项任务的结果。
            :param bool use_dict:
                此次分词是否使用用户词典。
            :param bool return_list:
                是否以list的形式将结果返回。默认为True, 如果设为False. 将以Sentence类的形式返回结果。
        '''
        #若输入的是字符串，转为一个元素的list
        if isinstance(sentence, str) and not isinstance(sentence, list):
            sentence = [sentence]
        elif isinstance(sentence, list):
            for s in sentence:
                if not isinstance(s, str):
                    raise ValueError(
                        "model can only parse string or list of string.")
        else:
            raise ValueError("model can only parse string or list of string.")

        if return_list is False and return_loc is False:
            return_loc = True
            print("return_list is False, return_loc is set to True")

        #去掉句子头尾空格以及行内空格
        for i, s in enumerate(sentence):
            s = "".join(s.split())
            sentence[i] = s.strip()

        if target not in [
                'CWS', 'POS', 'CWS-guwen', 'POS-guwen', 'NER', 'Parsing'
        ]:
            raise ValueError(
                "target can only be CWS, POS, CWS-guwen, POS-guwen, NER or Parsing."
            )

        if use_dict is True:
            task_class = self.style[target]
            if target == 'Parsing':
                task_class = self.style['POS']
            if target == 'CWS-guwen':
                task_class = self.style['POS-guwen']
            tag_seqs = []
            tag_dict = self.__get_tag_dict(task_class)
            important_tags = ['b', 'e', 'm', 's']
            for s in sentence:
                _, tag_seq = self.user_dict(s)
                t_tag_seq = torch.zeros([
                    len(tag_seq),
                    len(self.task_vocab_map[task_class].word2idx)
                ])
                for i in range(len(tag_seq)):
                    if tag_seq[i] in important_tags:
                        for tag in tag_dict[tag_seq[i]].values():
                            t_tag_seq[i][tag] = 1
                tag_seqs.append(t_tag_seq.tolist())
        else:
            tag_seqs = None

        dataloader, task_class = self.__preprocess_sentence(
            sentence, tag_seqs, target)

        if target in ['CWS', 'POS', 'CWS-guwen', 'POS-guwen', 'NER']:
            res = []
            seq_lens = []
            for idx, batch_samples in enumerate(dataloader):
                input_ids = batch_samples['input_ids'].to(self.device)
                attention_mask = batch_samples['attention_mask'].to(
                    self.device)
                seq_len = batch_samples['seq_len']
                task = batch_samples['task'].to(self.device)
                if tag_seqs is not None:
                    t_tag_seqs = batch_samples['tag_seqs'].to(self.device)
                else:
                    t_tag_seqs = None
                output = self.model(input_ids,
                                    attention_mask,
                                    task,
                                    tag_seqs=t_tag_seqs,
                                    user_dict_weight=self.user_dict_weight)
                res.extend(output['pred'].tolist())
                seq_lens.extend(seq_len.tolist())
            #将输出的一个batch的标签向量，逐句转换为分好的词以及每个词的属性。
            ans = []
            for i in range(len(sentence)):
                tags = self._to_label(res[i][:seq_lens[i]], task_class)
                ans.append(
                    self._get_list(sentence[i],
                                   tags,
                                   target,
                                   return_loc=return_loc))
        else:
            # 依存语法分析
            res = {'label_preds': [], 'head_preds': []}
            seq_lens = []
            for idx, batch_samples in enumerate(dataloader):
                input_ids = batch_samples['input_ids'].to(self.device)
                attention_mask = batch_samples['attention_mask'].to(
                    self.device)
                seq_len = batch_samples['seq_len'].to(self.device)
                task = batch_samples['task'].to(self.device)
                output = self.model(input_ids, attention_mask, task)
                res['label_preds'].extend(output['label_preds'].tolist())
                res['head_preds'].extend(output['head_preds'].tolist())
                seq_lens.extend(seq_len.tolist())
            #逐句进行解析
            ans = []
            for i in range(len(sentence)):
                #解析依存分析信息
                head_preds = res['head_preds'][i][1:seq_lens[i]]
                label_preds = self._to_label(
                    res['label_preds'][i][1:seq_lens[i]], task_class)
                ans.append(self._parsing(head_preds, label_preds, sentence[i]))

        if return_list:
            return ans

        #将结果转化为Sentence组成的列表
        answer = []
        for ans_ in ans:
            answer.append(Sentence(ans_, target))
        return answer
