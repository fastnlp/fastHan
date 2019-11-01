import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastNLP.modules import ConditionalRandomField, allowed_transitions
from torch.nn import init
from torch.nn.parameter import Parameter


class ProjectionLayer(nn.Module):
    def __init__(self, embedding_dim=768, output_dim=4,num_corpus=10,hidden_dim=256,dropout=0.1):
        super().__init__()
        
        self.embedding_dim=embedding_dim
        self.output_dim=output_dim
        self.num_corpus=num_corpus
        self.domain_embedding=nn.Embedding(self.num_corpus,hidden_dim)
        self.fc1=nn.Linear((hidden_dim+embedding_dim),hidden_dim)
        self.fc2=nn.Linear(hidden_dim,output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, words, corpus=None):
        if corpus is None:
            print('error, no corpus input')

        out=self.domain_embedding(corpus)
        
        out=out.unsqueeze(1)
        length=words.size()[1]
        out=out.expand(-1,length,-1)
        out=torch.cat((words,out),dim=2)
        out=self.fc1(out)
        out=F.relu(out)
        out=self.dropout(out)
        out=self.fc2(out)
        
        return {"pred":out}

class Bert_Proj_CRF(nn.Module):
    def __init__(self, embed, tag_vocab, encoding_type='bmes',embedding_dim=768,dropout=0.1):
        super().__init__()
        self.embed = embed
        self.pjl = ProjectionLayer(embedding_dim,len(tag_vocab),dropout=dropout)
        trans = allowed_transitions(tag_vocab, encoding_type=encoding_type, include_start_end=True)
        self.crf = ConditionalRandomField(len(tag_vocab), include_start_end_trans=True, allowed_transitions=trans)

    def _forward(self, words, target,corpus):
        mask = words.ne(0)
        words = self.embed(words)
        words = self.pjl(words,corpus)['pred']
        logits = F.log_softmax(words, dim=-1)
        if target is not None:
            loss = self.crf(logits, target, mask)
            return {'loss': loss}
        else:
            paths, _ = self.crf.viterbi_decode(logits, mask)
            return {'pred': paths}

    def forward(self, words, target,corpus=None):
        return self._forward(words, target,corpus)

    def predict(self, words,corpus=None):
        return self._forward(words, target=None, corpus=corpus)
