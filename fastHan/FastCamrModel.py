import os
import copy
import torch
import pandas as pd

from shutil import copyfile
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader, SequentialSampler, RandomSampler
from fastNLP.io.file_utils import cached_path
from transformers import BartForConditionalGeneration, BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from .model.camr_finetune_dataloader import data_collator, FastCAMR_Parsing_Loader
from .model.camr_restore import restore_camr, convert_camr_to_lines

MAX_LEN = 300
from transformers.utils import logging

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings('ignore')

class FastCAMR(object):
    """
        FastCAMR可处理:
            中文AMR任务, 依照中文AMR语料库CAMR1.0的标准。
    """
    def __init__(self, url=None):
        self.device = 'cpu'
        #获取模型的目录/下载模型
        if url is not None:
            model_dir = url
        else:
            model_dir = self._get_model()

        self.model_dir = model_dir
        # 加载模型
        self.model = BartForConditionalGeneration.from_pretrained(
            self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)

    def finetune(self,
                 data_path=None,
                 lr=1e-5,
                 n_epochs=1,
                 batch_size=8,
                 save=False,
                 save_url=None):
        '''
        令FastCAMR继续在CAMR任务上进行微调。
            1. 用户传入用于微调的数据集文件并设置超参数, FastCAMR自动加载并完成微调。
            2. 用户可在微调后保存模型参数,并在初始化时通过url加载之前微调过的模型。
            3. 进行微调的设备为当前FastCAMR的device。

        数据集文件的格式要依照中文AMR语料库CAMR1.0的格式, 如下所示。
        Example::
            # ::id export_amr.1322 ::2017-01-04
            # ::snt 这样 的 活动 还 有 什么 意义 呢 ？
            # ::wid x1_这样 x2_的 x3_活动 x4_还 x5_有 x6_什么 x7_意义 x8_呢 x9_？ x10_
            (x5/有-03 
                :mod()(x4/还) 
                :arg1()(x7/意义 
                    :mod()(x11/amr-unknown)) 
                :mode()(x2/的) 
                :mod-of(x12/的而)(x1/这样))

            
            # ::id export_amr.1327 ::2017-01-04
            # ::snt 并且 还 有 很多 高层 的 人物 哦 ！
            # ::wid x1_并且 x2_还 x3_有 x4_很多 x5_高层 x6_的 x7_人物 x8_哦 x9_！ x10_
            (x11/and 
                :op2(x1/并且)(x3/有-03 
                    :mod()(x2/还) 
                    :arg1()(x7/人物 
                        :mod-of(x6/的)(x5/高层) 
                        :quant()(x12/-))) 
                :mode()(x13/- 
                    :expressive()(x14/-)))
            ...
        相关格式的含义请详见中文AMR语料库CAMR1.0的标准。

        :param str data_path:   用于微调的数据集文件的路径。
        :param float lr:        微调的学习率。默认取1e-5。
        :param int n_epochs:    微调的迭代次数, 默认取1。
        :param int batch_size:  每个batch的数据数量, 默认为8。
        :param bool save:       是否保存微调后的模, 默认为False。
        :param str save_url:    若保存模型，则此值为保存模型的路径。
        '''
        assert (data_path)

        # 创建训练需要的相关参数
        dataset = FastCAMR_Parsing_Loader(data_path, self.tokenizer)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=RandomSampler(dataset),
                                collate_fn=data_collator)
        optimizer, scheduler = self._create_optimizer_and_scheduler(
            dataloader, n_epochs, lr)

        best_model = None
        best_eval_loss = 1e9
        for epoch in range(int(n_epochs)):
            self.model.train()
            tr_loss = 0
            nb_tr_steps = 0
            # 训练过程
            for step, inputs in enumerate(dataloader):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                with torch.set_grad_enabled(True):
                    loss = self.model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        decoder_input_ids=inputs['decoder_input_ids'],
                        labels=inputs['labels'])[0]
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    tr_loss += loss.item()
                    nb_tr_steps += 1
            loss = tr_loss / nb_tr_steps

            if loss < best_eval_loss:
                best_model = copy.deepcopy(self.model)
                best_eval_loss = loss

        self.model = best_model

        if save:
            if not os.path.exists(save_url):
                os.mkdir(save_url)
            print('saving model')
            self.model.save_pretrained(save_url)
            copyfile(os.path.join(self.model_dir, 'special_tokens_map.json'),
                     os.path.join(save_url, 'special_tokens_map.json'))
            copyfile(os.path.join(self.model_dir, 'tokenizer_config.json'),
                     os.path.join(save_url, 'tokenizer_config.json'))
            copyfile(os.path.join(self.model_dir, 'vocab.txt'),
                     os.path.join(save_url, 'vocab.txt'))
        return

    def _create_optimizer_and_scheduler(self, dataloader, n_epochs, lr):
        num_training_steps = len(dataloader) * n_epochs
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.1,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.1 * num_training_steps,
            num_training_steps=num_training_steps)
        return optimizer, scheduler

    def _get_model(self):
        """
            首先检查本地目录中是否已缓存模型，若没有缓存则下载。
        """
        url = 'http://download.fastnlp.top/fasthan/fastCAMR.zip'
        model_dir = cached_path(url, name='FastCAMR')
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

    def __restore_camr(self, lines, id_token_list):
        amr_lines = []
        for idx, line in enumerate(lines):
            line = line.strip()
            amr_list = restore_camr(line, id_token_list[idx])
            amr_lines.append(convert_camr_to_lines(''.join(amr_list)))
        return amr_lines

    def __preprocess_sentence(self, sentence):
        # 参照中文AMR语料库CAMR1.0的标准, 为句子中每个词汇设置对应的编号
        id_token_list = []
        for idx, sent in enumerate(sentence):
            id_token_dict = {}
            words = sent.split()
            word_idx = 1
            for word in words:
                if word != '':
                    id_token_dict[word_idx] = word
                    word_idx += 1
            id_token_list.append(id_token_dict)

        # 获取dataloader
        input_ids, attention_mask = [], []
        for idx, sent in enumerate(sentence):
            sent_tokenize_result = self.tokenizer(sent,
                                                  max_length=MAX_LEN,
                                                  padding='max_length',
                                                  truncation=True)
            input_ids.append(sent_tokenize_result['input_ids'])
            attention_mask.append(sent_tokenize_result['attention_mask'])
        amr_data = {"input_ids": input_ids, "attention_mask": attention_mask}
        amr_data = pd.DataFrame(amr_data)
        dataset = Dataset.from_pandas(amr_data, preserve_index=False)
        dataloader = DataLoader(dataset,
                                batch_size=1,
                                sampler=SequentialSampler(dataset),
                                collate_fn=data_collator)
        return id_token_list, dataloader

    def __call__(self, sentence):
        '''
            用户调用FastCAMR的接口函数。
            调用后会返回list类。

            :param str,list sentence:
                用于解析的输入, 可以是字符串形式的一个句子, 也可以是由字符串形式的句子组成的列表。
                输入的每个句子必须用空格进行分词, 可以使用fastHan中的分词功能首先对原始句子进行分词。
                每个句子的长度需要小于等于300。
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

        #去掉句子头尾空格
        for i, s in enumerate(sentence):
            sentence[i] = s.strip()

        id_token_list, dataloader = self.__preprocess_sentence(sentence)

        result_amrs = []
        for step, inputs in enumerate(dataloader):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            with torch.set_grad_enabled(False):
                generate_amrs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    num_beams=2,
                    length_penalty=2.0,
                    max_length=512,
                    min_length=0,
                    early_stopping=True,
                )
                generate_amrs = [
                    self.tokenizer.decode(g,
                                          skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
                    for g in generate_amrs
                ]
                # print(generate_amrs)
                result_amrs.extend(generate_amrs)

        formatted_amrs = self.__restore_camr(result_amrs, id_token_list)
        return formatted_amrs