# from multitask_model_normloss2
# 使用 wm来管理

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastNLP.modules.torch import MLP, ConditionalRandomField, allowed_transitions
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel

from .dependency_parsing_model import BertParser


# modified from https://github.com/THUDM/P-tuning-v2
class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self,
                 config,
                 num_tokens,
                 prefix_projection,
                 pre_seq_len,
                 prefix_hidden_size=500):
        super().__init__()
        self.prefix_projection = prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(num_tokens, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(
                    prefix_hidden_size,
                    config.num_hidden_layers * 2 * config.hidden_size))
        else:
            self.embedding = torch.nn.Embedding(
                num_tokens, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class MultiTaskModel(PreTrainedModel):
    def __init__(self,
                 encoder,
                 task_label_map,
                 all_tasks,
                 ensembledWeightManager,
                 prefix_projection=False,
                 pre_seq_len=6,
                 biaffine_task='Parsing-ctb9'):
        super().__init__(encoder.config)

        self.all_tasks = all_tasks
        self.task_label_map = task_label_map
        self.ensembledWeightManager = ensembledWeightManager
        # sequence_labeling
        self.biaffine_task = biaffine_task
        self.seq_label_classifier = nn.ModuleDict()
        self.crf = nn.ModuleDict()

        for task in task_label_map:
            if task == self.biaffine_task:
                continue

            label_num = len(task_label_map[task])
            labels = {i: task_label_map[task][i] for i in range(label_num)}

            self.seq_label_classifier[task] = MLP(
                [encoder.config.hidden_size, 512, label_num])
            self.crf[task] = ConditionalRandomField(
                num_tags=label_num,
                allowed_transitions=allowed_transitions(labels))
            self.crf[task].trans_m.data *= 0

        self.parser = BertParser(
            num_label=len(task_label_map[self.biaffine_task]),
            embed_size=encoder.config.hidden_size,
            app_index=task_label_map[self.biaffine_task].index('app'))
        self.dropout = nn.Dropout(encoder.config.hidden_dropout_prob)
        self.encoder = encoder

        # prefix tuning
        self.pre_seq_len = pre_seq_len
        self.build_prefix_map()
        self.prefix_encoder = PrefixEncoder(
            config=encoder.config,
            prefix_projection=prefix_projection,
            num_tokens=self.num_tokens,
            pre_seq_len=self.pre_seq_len)

    def build_prefix_map(self):
        length = self.pre_seq_len // 2
        idx = 0
        prefix_map = dict()
        macro_map = dict()
        for task in self.all_tasks:
            macro_task, _ = task.split('-')
            if macro_task not in macro_map:
                macro_map[macro_task] = [idx + i for i in range(length)]
                idx += length

        for task in self.all_tasks:
            macro_task, _ = task.split('-')
            prefix_map[task] = torch.LongTensor(
                macro_map[macro_task] + [idx + i for i in range(length)])
            idx += length
        self.prefix_map = prefix_map
        self.num_tokens = idx

    def get_prompt(self, task, batch_size):
        prefix_tokens = self.prefix_map[task]
        prefix_tokens = prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(
            self.encoder.device)
        past_key_values = self.prefix_encoder(prefix_tokens)

        past_key_values = past_key_values.view(
            batch_size, self.pre_seq_len,
            self.encoder.config.num_hidden_layers * 2,
            self.encoder.config.num_attention_heads,
            self.encoder.config.hidden_size //
            self.encoder.config.num_attention_heads)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def __get_ud_diff(self, feats, tag_seqs, user_dict_weight):
        diff = torch.max(feats, dim=2)[0] - torch.mean(feats, dim=2)
        diff = diff.unsqueeze(dim=-1)
        diff = diff.expand(-1, -1, tag_seqs.size()[-1])
        diff = tag_seqs * diff * user_dict_weight
        return diff

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        task=None,
        labels=None,
        heads=None,
        tag_seqs=None,
        user_dict_weight=0.05,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """

        task = task.tolist()[0]
        task = self.all_tasks[task]
    
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(task=task, batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(
            self.encoder.device)
        prefix_attention_mask = torch.cat(
            (prefix_attention_mask, attention_mask), dim=1)

        outputs = self.encoder(input_ids,
                               attention_mask=prefix_attention_mask,
                               past_key_values=past_key_values)

        feats = outputs[0]
        feats = self.dropout(feats)

        # 生成不考虑cls和sep的mask
        #if task==self.biaffine_task:
        seq_len_diff = 2
        #else:
        #    seq_len_diff=1
        seq_len = attention_mask.sum(dim=-1) - seq_len_diff
        broad_cast_seq_len = torch.arange(attention_mask.shape[1] -
                                          seq_len_diff).expand(
                                              attention_mask.shape[0],
                                              -1).to(seq_len.device)
        mask = broad_cast_seq_len < seq_len.unsqueeze(1)

        # dependency parsing
        # 需要去掉cls和sep的影响
        if task == self.biaffine_task:
            feats = feats[:, 1:-1]
            output = self.parser.forward(feats=feats,
                                         mask=mask,
                                         gold_heads=heads,
                                         char_labels=labels)
        # 其他序列标注任务
        else:
            logits = self.seq_label_classifier[task](feats)
            if self.training:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, len(self.task_label_map[task])),
                    labels.view(-1))
                output = {
                    'loss': loss,
                    'logits': logits,
                }
            else:
                # 预测阶段，利用crf模块中集成的维特比解码来预测
                probs = logits[:, 1:-1]
                if tag_seqs is not None:
                    diff = self.__get_ud_diff(probs, tag_seqs, user_dict_weight)
                    probs = probs + diff
                paths, scores = self.crf[task].viterbi_decode(logits=probs,
                                                              mask=mask)
                paths[mask == 0] = -100
                output = {
                    'pred': paths,
                    'logits': logits,
                }

        if self.training:
            if task == self.biaffine_task:
                loss_weight = output['label_loss']
            else:
                loss_weight = output['loss']
            self.ensembledWeightManager.update(task=task,
                                               loss=float(loss_weight))
            weight = self.ensembledWeightManager.get(task)
            output['loss'] = output['loss'] * weight

        return output