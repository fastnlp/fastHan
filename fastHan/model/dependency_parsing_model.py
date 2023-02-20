import torch
import torch.nn.functional as F
from fastNLP.models.torch.biaffine_parser import (ArcBiaffine, BiaffineParser,
                                                  LabelBilinear)
from fastNLP.modules.torch.dropout import TimestepDropout
from torch import nn
from transformers import PreTrainedModel


# modified from https://github.com/yhcc/JointCwsParser/blob/master/models/BertParser.py
class BertParser(BiaffineParser):
    def __init__(self,
                 num_label,
                 embed_size=768,
                 arc_mlp_size=500,
                 label_mlp_size=100,
                 dropout=0.1,
                 use_greedy_infer=False,
                 app_index=0):
        super(BiaffineParser, self).__init__()

        self.embed_size = embed_size
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_size, arc_mlp_size * 2 + label_mlp_size * 2),
            nn.LeakyReLU(0.1),
            TimestepDropout(p=dropout),
        )
        self.arc_mlp_size = arc_mlp_size
        self.label_mlp_size = label_mlp_size
        self.arc_predictor = ArcBiaffine(arc_mlp_size, bias=True)
        self.label_predictor = LabelBilinear(label_mlp_size,
                                             label_mlp_size,
                                             num_label,
                                             bias=True)
        self.use_greedy_infer = use_greedy_infer
        self.reset_parameters()

        self.app_index = app_index
        self.num_label = num_label
        if self.app_index != 0:
            raise ValueError("现在app_index必须等于0")

        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for name, m in self.named_modules():
            if hasattr(m, 'reset_parameters') or hasattr(m, 'init_param'):
                pass
            else:
                for p in m.parameters():
                    if len(p.size()) > 1:
                        nn.init.xavier_normal_(p, gain=0.1)
                    else:
                        nn.init.uniform_(p, -0.1, 0.1)

    def forward(self, feats, mask, gold_heads=None, char_labels=None):
        # 此处的mask与attention_mask不同，需要去除sep与cls

        batch_size = feats.shape[0]
        max_len = feats.shape[1]

        feats = self.dropout(feats)
        feats = self.mlp(feats)
        arc_sz, label_sz = self.arc_mlp_size, self.label_mlp_size
        arc_dep, arc_head = feats[:, :, :arc_sz], feats[:, :,
                                                        arc_sz:2 * arc_sz]
        label_dep, label_head = feats[:, :, 2 * arc_sz:2 * arc_sz +
                                      label_sz], feats[:, :,
                                                       2 * arc_sz + label_sz:]

        arc_pred = self.arc_predictor(arc_head, arc_dep)  # [N, L, L]

        if self.training:
            assert gold_heads is not None
            head_pred = None
            heads = gold_heads
        else:
            heads = self.mst_decoder(arc_pred, mask)
            head_pred = heads

        # 将pad的-100替换为-1，以免heads作为矩阵索引的时候报错
        padded_heads = torch.clone(heads)
        padded_heads[padded_heads == -100] = -1

        batch_range = torch.arange(start=0,
                                   end=batch_size,
                                   dtype=torch.long,
                                   device=mask.device).unsqueeze(1)
        label_head = label_head[batch_range, padded_heads].contiguous()
        label_pred = self.label_predictor(label_head,
                                          label_dep)  # [N, max_len, num_label]
        # 这里限制一下，只有当head为下一个时，才能预测app这个label
        arange_index = torch.arange(1, max_len+1, dtype=torch.long, device=mask.device).unsqueeze(0)\
            .repeat(batch_size, 1) # batch_size x max_len

        app_masks = heads.ne(
            arange_index)  #  batch_size x max_len, 为1的位置不可以预测app
        app_masks = app_masks.unsqueeze(2).repeat(1, 1, self.num_label)
        app_masks[:, :, 1:] = 0
        label_pred = label_pred.masked_fill(app_masks, float('-inf'))

        if self.training:
            arc_loss, label_loss = self.loss(arc_pred, label_pred, gold_heads,
                                             char_labels, mask)
            res_dict = {
                'loss': arc_loss + label_loss,
                'arc_loss': arc_loss,
                'label_loss': label_loss
            }
        else:
            res_dict = {
                'label_preds': label_pred.max(2)[1],
                'head_preds': head_pred
            }

        return res_dict

    @staticmethod
    def loss(arc_pred, label_pred, arc_true, label_true, mask):
        """
        Compute loss.
        :param arc_pred: [batch_size, seq_len, seq_len]
        :param label_pred: [batch_size, seq_len, n_tags]
        :param arc_true: [batch_size, seq_len]
        :param label_true: [batch_size, seq_len]
        :param mask: [batch_size, seq_len]
        :return: loss value
        """

        batch_size, seq_len, _ = arc_pred.shape
        flip_mask = (mask == 0)
        # _arc_pred = arc_pred.clone()
        _arc_pred = arc_pred.masked_fill(flip_mask.unsqueeze(1), -float('inf'))

        arc_true.data[:, 0].fill_(-100)
        label_true.data[:, 0].fill_(-100)

        arc_nll = F.cross_entropy(_arc_pred.view(-1, seq_len),
                                  arc_true.view(-1),
                                  ignore_index=-100)
        label_nll = F.cross_entropy(label_pred.view(-1, label_pred.size(-1)),
                                    label_true.view(-1),
                                    ignore_index=-100)

        return arc_nll, label_nll


class DependencyParsingModel(PreTrainedModel):
    def __init__(self, encoder, config, labels):
        super().__init__(config)

        label_num = len(labels)

        self.num_labels = label_num

        self.encoder = encoder
        self.parser = BertParser(num_label=self.num_labels,
                                 embed_size=config.hidden_size,
                                 app_index=labels.index('app'))

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                heads=None):
        # 生成不考虑cls和sep的mask
        seq_len = attention_mask.sum(dim=-1) - 2
        broad_cast_seq_len = torch.arange(attention_mask.shape[1] - 2).expand(
            attention_mask.shape[0], -1).to(seq_len.device)
        mask = broad_cast_seq_len < seq_len.unsqueeze(1)

        outputs = self.encoder(input_ids, attention_mask=attention_mask)

        feats = outputs[0]
        feats = feats[:, 1:-1]

        return self.parser.forward(feats=feats,
                                   mask=mask,
                                   gold_heads=heads,
                                   char_labels=labels)
