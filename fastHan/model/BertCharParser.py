


from fastNLP.models.biaffine_parser import BiaffineParser
from fastNLP.models.biaffine_parser import ArcBiaffine, LabelBilinear

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fastNLP.modules.dropout import TimestepDropout
from fastNLP.modules.encoder.variational_rnn import VarLSTM
from fastNLP import seq_len_to_mask
from fastNLP.embeddings import Embedding


def drop_input_independent(word_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.new(batch_size, seq_length).fill_(1 - dropout_emb)
    word_masks = torch.bernoulli(word_masks)
    word_masks = word_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks

    return word_embeddings


class CharBiaffineParser(BiaffineParser):
    def __init__(self,
                        vector_size,
                        num_label,
                        rnn_layers=3,
                        arc_mlp_size=500,
                        label_mlp_size=100,
                        dropout=0.3,
                        use_greedy_infer=False):


        super(BiaffineParser, self).__init__()
        rnn_out_size = vector_size
        self.timestep_drop = TimestepDropout(dropout)

        self.mlp = nn.Sequential(nn.Linear(rnn_out_size, arc_mlp_size * 2 + label_mlp_size * 2),
                                          nn.LeakyReLU(0.1),
                                          TimestepDropout(p=dropout),)
        self.arc_mlp_size = arc_mlp_size
        self.label_mlp_size = label_mlp_size
        self.arc_predictor = ArcBiaffine(arc_mlp_size, bias=True)
        self.label_predictor = LabelBilinear(label_mlp_size, label_mlp_size, num_label, bias=True)
        self.use_greedy_infer = use_greedy_infer
        self.reset_parameters()
        self.dropout = dropout


        self.num_label = num_label


    def reset_parameters(self):
        for name, m in self.named_modules():
            if 'embed' in name:
                pass
            elif hasattr(m, 'reset_parameters') or hasattr(m, 'init_param'):
                pass
            else:
                for p in m.parameters():
                    if len(p.size())>1:
                        nn.init.xavier_normal_(p, gain=0.1)
                    else:
                        nn.init.uniform_(p, -0.1, 0.1)

    def forward(self, feats, seq_lens, gold_heads=None):
        """
        max_len是包含root的
        :param chars: batch_size x max_len
        :param ngrams: batch_size x max_len*ngram_per_char
        :param seq_lens: batch_size
        :param gold_heads: batch_size x max_len
        :param pre_chars: batch_size x max_len
        :param pre_ngrams: batch_size x max_len*ngram_per_char
        :return dict: parsing results
            arc_pred: [batch_size, seq_len, seq_len]
            label_pred: [batch_size, seq_len, seq_len]
            mask: [batch_size, seq_len]
            head_pred: [batch_size, seq_len] if gold_heads is not provided, predicting the heads
        """
        # prepare embeddings
        batch_size,seq_len,_ = feats.shape
        # print('forward {} {}'.format(batch_size, seq_len))

        # get sequence mask
        mask = seq_len_to_mask(seq_lens).long()

        # for arc biaffine
        # mlp, reduce dim
        feat = self.mlp(feats)
        arc_sz, label_sz = self.arc_mlp_size, self.label_mlp_size
        arc_dep, arc_head = feat[:,:,:arc_sz], feat[:,:,arc_sz:2*arc_sz]
        label_dep, label_head = feat[:,:,2*arc_sz:2*arc_sz+label_sz], feat[:,:,2*arc_sz+label_sz:]

        # biaffine arc classifier
        arc_pred = self.arc_predictor(arc_head, arc_dep) # [N, L, L]
        # use gold or predicted arc to predict label
        if gold_heads is None or not self.training:
            # use greedy decoding in training
            if self.training or self.use_greedy_infer:
                heads = self.greedy_decoder(arc_pred, mask)
            else:
                heads = self.mst_decoder(arc_pred, mask)
            head_pred = heads
        else:
            assert self.training # must be training mode
            if gold_heads is None:
                heads = self.greedy_decoder(arc_pred, mask)
                head_pred = heads
            else:
                head_pred = None
                heads = gold_heads
        # heads: batch_size x max_len

        batch_range = torch.arange(start=0, end=batch_size, dtype=torch.long, device=feats.device).unsqueeze(1)
        label_head = label_head[batch_range, heads].contiguous()
        label_pred = self.label_predictor(label_head, label_dep) # [N, max_len, num_label]

        res_dict = {'arc_pred': arc_pred, 'label_pred': label_pred, 'mask': mask}
        if head_pred is not None:
            res_dict['head_pred'] = head_pred
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

        arc_true.data[:, 0].fill_(-1)
        label_true.data[:, 0].fill_(-1)

        arc_nll = F.cross_entropy(_arc_pred.view(-1, seq_len), arc_true.view(-1), ignore_index=-1)
        label_nll = F.cross_entropy(label_pred.view(-1, label_pred.size(-1)), label_true.view(-1), ignore_index=-1)
        
        return arc_nll + label_nll

class BertCharParser(nn.Module):
    def __init__(self,  app_index,
                        vector_size,
                        num_label,
                        rnn_layers=3,
                        arc_mlp_size=500,
                        label_mlp_size=100,
                        dropout=0.3,
                        use_greedy_infer=False):
        super().__init__()

        self.parser = CharBiaffineParser(
                        vector_size,
                        num_label,
                        rnn_layers,
                        arc_mlp_size,
                        label_mlp_size,
                        dropout,
                        use_greedy_infer)
        self.app_index=app_index

    def forward(self, feats, seq_lens, char_heads, char_labels):
        res_dict = self.parser(feats, seq_lens, gold_heads=char_heads)
        arc_pred = res_dict['arc_pred']
        label_pred = res_dict['label_pred']
        masks = res_dict['mask']
        loss = self.parser.loss(arc_pred, label_pred, char_heads, char_labels, masks)
        return {'loss': loss}

    def predict(self, feats, seq_lens):
        res = self.parser(feats, seq_lens, gold_heads=None)
        output = {}
        output['head_preds'] = res.pop('head_pred')

        size=res['label_pred'].size()
        res['label_pred']=res['label_pred'].reshape(-1,size[-1])
        res['label_pred'][:,self.app_index]=-float('inf')
        res['label_pred']=res['label_pred'].reshape(size)

        _, label_pred = res.pop('label_pred').max(2)
        output['label_preds'] = label_pred
        return output