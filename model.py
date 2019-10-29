import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastNLP.modules import ConditionalRandomField, allowed_transitions
from torch.nn import init
from torch.nn.parameter import Parameter


class ProjectionLayer(nn.Module):
    def __init__(self, embedding_dim=768, output_dim=4,num_corpus=10):
        super().__init__()
        
        self.embedding_dim=embedding_dim
        self.output_dim=output_dim
        self.num_corpus=num_corpus
        self.domain_projection=Parameter(torch.Tensor(num_corpus,embedding_dim,output_dim))
        self.bias=Parameter(torch.Tensor(num_corpus,output_dim))
        self.shared_projection = nn.Linear(embedding_dim, output_dim)
        #self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.domain_projection, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.domain_projection)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, words, corpus=None):
        out1=self.shared_projection(words)
        if corpus is None:
            return {"pred":out1}
        
        B=words.size()[0]
        A=torch.index_select(self.domain_projection,0,corpus)
        b=torch.index_select(self.bias,0,corpus).unsqueeze(1)
        # A.size: B*emb*out
        out2=words.bmm(A)
        out2=out2+b
        
        return {"pred":out1+out2}

class Bert_Proj_CRF(nn.Module):
    def __init__(self, embed, tag_vocab, encoding_type='bmes',embedding_dim=768):
        super().__init__()
        self.embed = embed
        self.pjl = ProjectionLayer(embedding_dim,len(tag_vocab))
        self.fc=nn.Linear(embedding_dim,len(tag_vocab))
        trans = allowed_transitions(tag_vocab, encoding_type=encoding_type, include_start_end=True)
        self.crf = ConditionalRandomField(len(tag_vocab), include_start_end_trans=True, allowed_transitions=trans)

    def _forward(self, words, target,corpus):
        mask = words.ne(0)
        words = self.embed(words)
        words = self.pjl(words,corpus)['pred']
        #words = self.fc(words)
        #print(torch.mean(words))
        logits = F.log_softmax(words, dim=-1)
        if target is not None:
            loss = self.crf(logits, target, mask)
            return {'loss': loss}
        else:
            paths, _ = self.crf.viterbi_decode(logits, mask)
            return {'pred': paths}

    def forward(self, words, target,corpus=None):
        return self._forward(words, target,corpus)

    def predict(self, words):
        return self._forward(words, None, None)
