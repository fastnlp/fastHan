from fastNLP.core.metrics import Metric
import torch

# modified from https://github.com/yhcc/JointCwsParser
class SegAppCharParseF1Metric(Metric):
    #
    def __init__(self,pun_index,app_index=0):
        super().__init__()
        self.app_index = app_index
        self.pun_index = pun_index

        self.parse_head_tp = 0
        self.parse_label_tp = 0
        self.rec_tol = 0
        self.pre_tol = 0

    def get_word_pairs(self,head_preds,label_preds,seq_lens,pun_masks):
        # 去掉root
        head_preds = head_preds[:, 1:].tolist()
        label_preds = label_preds[:, 1:].tolist()
        seq_lens = (seq_lens - 1).tolist()

        head_dep_tuples=[]
        head_label_dep_tuples = []

        for b in range(len(head_preds)):
            seq_len = seq_lens[b]
            head_pred = head_preds[b][:seq_len]
            label_pred = label_preds[b][:seq_len]

            words = [] # 存放[word_start, word_end)，相对起始位置，不考虑root
            heads = []
            labels = []
            ranges = []  # 对应该char是第几个word，长度是seq_len+1
            word_idx = 0
            word_start_idx = 0
            for idx, (label, head) in enumerate(zip(label_pred, head_pred)):
                ranges.append(word_idx)
                if label == self.app_index:
                    pass
                else:
                    labels.append(label)
                    heads.append(head)
                    words.append((word_start_idx, idx+1))
                    word_start_idx = idx+1
                    word_idx += 1

            head_dep_tuple = [] # head在前面
            head_label_dep_tuple = []
            for idx, head in enumerate(heads):
                span = words[idx]
                if span[0]==span[1]-1 and pun_masks[b, span[0]]:
                    continue  # exclude punctuations
                if head == 0:
                    head_dep_tuple.append((('root', words[idx])))
                    head_label_dep_tuple.append(('root', labels[idx], words[idx]))
                else:
                    head_word_idx = ranges[head-1]
                    head_word_span = words[head_word_idx]
                    head_dep_tuple.append(((head_word_span, words[idx])))
                    head_label_dep_tuple.append((head_word_span, labels[idx], words[idx]))
            head_dep_tuples.append(head_dep_tuple)
            head_label_dep_tuples.append(head_label_dep_tuple)
        
        return head_dep_tuples,head_label_dep_tuples


    # def update(self, gold_word_pairs, gold_label_word_pairs, head_preds, label_preds, seq_len,
    #              pun_masks):
    def update(self, labels,heads,head_preds, label_preds, seq_len):
        """

        max_len是不包含root的character的长度
        :param gold_word_pairs: List[List[((head_start, head_end), (dep_start, dep_end)), ...]], batch_size
        :param gold_label_word_pairs: List[List[((head_start, head_end), label, (dep_start, dep_end)), ...]], batch_size
        :param head_preds: batch_size x max_len
        :param label_preds: batch_size x max_len
        :param seq_lens:
        :return:
        """
        pun_masks=(labels==self.pun_index).long()
        pun_masks=pun_masks[:,1:]

        head_dep_tuples,head_label_dep_tuples=self.get_word_pairs(head_preds,label_preds,seq_len,pun_masks)
        gold_head_dep_tuples,gold_head_label_dep_tuples=self.get_word_pairs(heads,labels,seq_len,pun_masks)

        for b in range(seq_len.shape[0]):
            head_dep_tuple=head_dep_tuples[b]
            head_label_dep_tuple=head_label_dep_tuples[b]
            gold_head_dep_tuple=gold_head_dep_tuples[b]
            gold_head_label_dep_tuple=gold_head_label_dep_tuples[b]


            for head_dep, head_label_dep in zip(head_dep_tuple, head_label_dep_tuple):
                if head_dep in gold_head_dep_tuple:
                    self.parse_head_tp += 1
                if head_label_dep in gold_head_label_dep_tuple:
                    self.parse_label_tp += 1
            self.pre_tol += len(head_dep_tuple)
            self.rec_tol += len(gold_head_dep_tuple)

    def get_metric(self, reset=True):
        u_p = self.parse_head_tp / self.pre_tol
        u_r = self.parse_head_tp / self.rec_tol
        u_f = 2*u_p*u_r/(1e-6 + u_p + u_r)
        l_p = self.parse_label_tp / self.pre_tol
        l_r = self.parse_label_tp / self.rec_tol
        l_f = 2*l_p*l_r/(1e-6 + l_p + l_r)

        if reset:
            self.parse_head_tp = 0
            self.parse_label_tp = 0
            self.rec_tol = 0
            self.pre_tol = 0

        return {'u_f1': round(u_f, 4), 'u_p': round(u_p, 4), 'u_r/uas':round(u_r, 4),
                'f': round(l_f, 4), 'l_p': round(l_p, 4), 'l_r/las': round(l_r, 4)}


class CWSMetric(Metric):
    def __init__(self, app_index=0):
        super().__init__()
        self.app_index = app_index
        self.pre = 0
        self.rec = 0
        self.tp = 0


    def label_to_seg(self,labels,seq_lens):
        segs=torch.zeros_like(labels)[:,1:]
        masks=torch.zeros_like(labels)[:,1:]

        seq_lens=(seq_lens-1).tolist()
        # [:,1:]是为了剔除root结点
        for idx,label in enumerate(labels[:, 1:].tolist()):
            seq_len=seq_lens[idx]
            label=label[:seq_len]
            word_len = 0

            for i,l in enumerate(label):
                if l==self.app_index and i!=len(label)-1:
                    word_len+=1
                else:
                    segs[idx,i]=word_len
                    masks[idx,i]=1
                    word_len=0
        return segs,masks


    def update(self, labels, label_preds, seq_len):
        """
        :param label_preds: batch_size x max_len
        :param seq_len: batch_size
        :return:
        """

        seg_targets,seg_masks=self.label_to_seg(labels,seq_len)
        pred_segs,pred_masks=self.label_to_seg(label_preds,seq_len)


        right_mask = seg_targets.eq(pred_segs) # 对长度的预测一致
        self.rec += seg_masks.sum().item()
        self.pre += pred_masks.sum().item()
        # 且pred和target在同一个地方有值
        self.tp += (right_mask.__and__(pred_masks.bool().__and__(seg_masks.bool()))).sum().item()

    def get_metric(self, reset=True):
        res = {}
        res['rec'] = round(self.tp/(self.rec+1e-6), 4)
        res['pre'] = round(self.tp/(self.pre+1e-6), 4)
        res['f1'] = round(2*res['rec']*res['pre']/(res['pre'] + res['rec'] + 1e-6), 4)

        if reset:
            self.pre = 0
            self.rec = 0
            self.tp = 0

        return res