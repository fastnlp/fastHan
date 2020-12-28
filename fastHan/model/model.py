import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastNLP import CrossEntropyLoss
from fastNLP.modules import MLP, ConditionalRandomField, allowed_transitions
from torch.nn import init
from torch.nn.parameter import Parameter
from .BertCharParser import BertCharParser


class CharModel(nn.Module):
    def __init__(self, embed,label_vocab,pos_idx=31,
                Parsing_rnn_layers=3, Parsing_arc_mlp_size=500,
                Parsing_label_mlp_size=100,Parsing_use_greedy_infer=False,
                encoding_type='bmeso',embedding_dim=768,dropout=0.1,use_pos_embedding=True,
                use_average=True):
        super().__init__()
        self.embed = embed
        self.use_pos_embedding=use_pos_embedding
        self.use_average=use_average
        self.label_vocab=label_vocab
        self.pos_idx=pos_idx
        self.user_dict_weight=0.05
        embedding_dim_1=512
        embedding_dim_2=256
        
        
        self.layers_map={'CWS':'-1','POS':'-1','Parsing':'-1','NER':'-1'}
        #NER
        self.ner_linear=nn.Linear(embedding_dim,len(label_vocab['NER']))
        trans = allowed_transitions(label_vocab['NER'], encoding_type='bmeso', include_start_end=True)
        self.ner_crf = ConditionalRandomField(len(label_vocab['NER']), include_start_end_trans=True, allowed_transitions=trans)

        #parsing
        self.biaffine_parser=BertCharParser(
                    app_index=self.label_vocab['Parsing'].to_index('APP'),
                    vector_size=768,
                    num_label=len(label_vocab['Parsing']),
                    rnn_layers=Parsing_rnn_layers,
                    arc_mlp_size=Parsing_arc_mlp_size,
                    label_mlp_size=Parsing_label_mlp_size,
                    dropout=dropout,
                    use_greedy_infer=Parsing_use_greedy_infer)
        
        if self.use_pos_embedding:
            self.pos_embedding=nn.Embedding(len(self.label_vocab['pos']),embedding_dim, padding_idx=0)
        
        
        self.loss=CrossEntropyLoss(padding_idx=0)

        #CWS
        self.cws_mlp=MLP([embedding_dim, embedding_dim_1,embedding_dim_2, len(label_vocab['CWS'])], 'relu', output_activation=None)
        trans=allowed_transitions(label_vocab['CWS'],include_start_end=True)
        self.cws_crf = ConditionalRandomField(len(label_vocab['CWS']), include_start_end_trans=True, allowed_transitions=trans)

        #POS
        self.pos_mlp=MLP([embedding_dim, embedding_dim_1,embedding_dim_2, len(label_vocab['POS'])], 'relu', output_activation=None)
        trans=allowed_transitions(label_vocab['POS'],include_start_end=True)
        self.pos_crf = ConditionalRandomField(len(label_vocab['POS']), include_start_end_trans=True, allowed_transitions=trans)

    def _generate_embedding(self,feats,word_lens,seq_len,pos):
        device=feats.device
        new_feats=[]
        batch_size=feats.size()[0]
        sentence_length=feats.size()[1]
        if self.use_average==False:
            for i in range(batch_size):
                new_feats.append(torch.index_select(feats[i],0,word_lens[i]))
            new_feats=torch.stack(new_feats,0)
        else:
            for i in range(batch_size):
                feats_for_one_sample=[]
                for j in range(word_lens.size()[1]):
                    if word_lens[i][j]==0 and j!=0:
                        feats_for_one_word=torch.zeros(feats.size()[-1]).to(device)
                    else:
                        if j==word_lens.size()[1]-1 or word_lens[i][j+1]==0:
                            index=range(word_lens[i][j],seq_len[i])
                        else:
                            index=range(word_lens[i][j],word_lens[i][j+1])
                        
                        index=torch.tensor(index)
                        index=index.to(device)
                        feats_for_one_word=torch.index_select(feats[i],0,index)
                        feats_for_one_word=torch.mean(feats_for_one_word,dim=0)
                        feats_for_one_word=feats_for_one_word.to(device)
                    feats_for_one_sample.append(feats_for_one_word)
                feats_for_one_sample=torch.stack(feats_for_one_sample,dim=0)
                new_feats.append(feats_for_one_sample)
            new_feats=torch.stack(new_feats,0)
        if self.use_pos_embedding:
            pos_feats=self.pos_embedding(pos)
            new_feats=new_feats+pos_feats
        return new_feats
    
    def _generate_from_pos(self,paths,seq_len):
        device=paths.device
        word_lens=[]
        batch_size=paths.size()[0]
        new_seq_len=[]
        batch_pos=[]
        for i in range(batch_size):
            word_len=[]
            pos=[]
            for j in range(seq_len[i]):
                tag=paths[i][j]
                tag=self.label_vocab['POS'].to_word(int(tag))
                if tag.startswith('<'):
                    continue
                tag1,tag2=tag.split('-')
                tag2=self.label_vocab['pos'].to_index(tag2)
                if tag1=='S' or tag1=='B':
                    word_len.append(j)
                    pos.append(tag2)
            if len(pos)==1:
                word_len.append(seq_len[i]-1)
                pos.append(tag2)
            new_seq_len.append(len(pos))
            word_lens.append(word_len)
            batch_pos.append(pos)
        max_len=max(new_seq_len)
        for i in range(batch_size):
            word_lens[i]=word_lens[i]+[0]*(max_len-new_seq_len[i])
            batch_pos[i]=batch_pos[i]+[0]*(max_len-new_seq_len[i])
        word_lens=torch.tensor(word_lens,device=device)
        batch_pos=torch.tensor(batch_pos,device=device)
        new_seq_len=torch.tensor(new_seq_len,device=device)
        return word_lens,batch_pos,new_seq_len
   
    def _decode_parsing(self,dep_head,dep_label,seq_len,seq_len_for_wordlist,word_lens):
        device=dep_head.device
        heads=[]
        labels=[]
        batch_size=dep_head.size()[0]
        app_index=self.label_vocab['Parsing'].to_index('APP')
            
        max_len=seq_len.max()
        for i in range(batch_size):
            head=list(range(1,seq_len[i]+1))
            label=[app_index]*int(seq_len[i])
            head[0]=0
                
            for j in range(1,seq_len_for_wordlist[i]):
                if j+1==seq_len_for_wordlist[i]:
                    idx=seq_len[i]-1
                else:
                    idx=word_lens[i][j+1]-1
                
                label[idx]=int(dep_label[i][j])
                root=dep_head[i][j]
                if root>=seq_len_for_wordlist[i]-1:
                    head[idx]=int(seq_len[i]-1)
                else:
                    try:
                        head[idx]=int(word_lens[i][root+1]-1)
                    except:
                        print(len(head),idx,word_lens.size(),i,root)
            
            head=head+[0]*int(max_len-seq_len[i])
            label=label+[0]*int(max_len-seq_len[i])
            
            heads.append(head)
            labels.append(label)
        heads=torch.tensor(heads,device=device)
        labels=torch.tensor(labels,device=device)
        
        return heads,labels
    
    def forward(self, chars,seq_len,task_class,target,
                seq_len_for_wordlist=None,dep_head=None,dep_label=None,pos=None,word_lens=None):
        task=task_class[0]
        self.current_task=task
        mask = chars.ne(0)

        layers=self.layers_map[task]
        feats=self.embed(chars,layers)
        
        if task=='Parsing':
            parsing_feats=self._generate_embedding(feats,word_lens,seq_len,pos)
            loss_parsing=self.biaffine_parser(parsing_feats,seq_len_for_wordlist,dep_head,dep_label)
            
            return loss_parsing
        

        if task=='NER':
            #?需要relu吗
            feats=F.relu(self.ner_linear(feats))
            logits = F.log_softmax(feats, dim=-1)
            loss = self.ner_crf(logits, target, mask)
            return {'loss':loss}
        
        if task=='CWS':
            feats=self.cws_mlp(feats)
            logits=F.log_softmax(feats, dim=-1)
            loss=self.cws_crf(logits, target, mask)
            #loss=self.loss.get_loss(feats,target,seq_len)
            return {'loss':loss}


        if task=='POS':
            feats=self.pos_mlp(feats)
            logits=F.log_softmax(feats, dim=-1)
            loss=self.pos_crf(logits, target, mask)
            #loss=self.loss.get_loss(feats,target,seq_len)
            return {'loss':loss}

            



    def predict(self, chars,seq_len,task_class,tag_seqs=None):
        task=task_class[0]
        mask = chars.ne(0)
        layers=self.layers_map[task]
        feats=self.embed(chars,layers)
        
        if task=='Parsing':
            for sample in chars:
                sample[0]=self.pos_idx
            pos_feats=self.embed(chars,self.layers_map['POS'])
            pos_feats=self.pos_mlp(pos_feats)
            logits = F.log_softmax(pos_feats, dim=-1)
            paths, _ = self.pos_crf.viterbi_decode(logits, mask)
            #paths=pos_feats.max(dim=-1)[1]

            
            word_lens,batch_pos,seq_len_for_wordlist=self._generate_from_pos(paths,seq_len)
            parsing_feats=self._generate_embedding(feats,word_lens,seq_len,batch_pos)
            answer=self.biaffine_parser.predict(parsing_feats,seq_len_for_wordlist)
            head_preds=answer['head_preds']
            label_preds=answer['label_preds']
            heads,labels=self._decode_parsing(head_preds,label_preds,seq_len,seq_len_for_wordlist,word_lens)
            
            return {'head_preds':heads,'label_preds':labels,'pred':paths}

        if task=='CWS':
            feats=self.cws_mlp(feats)
            if tag_seqs is not None:
                diff=torch.max(feats,dim=2)[0]-torch.mean(feats,dim=2)
                diff=diff.unsqueeze(dim=-1)
                diff=diff.expand(-1,-1,len(self.label_vocab['CWS']))
                diff=tag_seqs*diff*self.user_dict_weight
                feats=feats+diff
            logits = F.log_softmax(feats, dim=-1)
            paths, _ = self.cws_crf.viterbi_decode(logits, mask)
            #paths=feats.max(dim=-1)[1]
            return {'pred': paths}

        if task=='POS':
            feats=self.pos_mlp(feats)
            logits = F.log_softmax(feats, dim=-1)
            paths, _ = self.pos_crf.viterbi_decode(logits, mask)
            #paths=feats.max(dim=-1)[1]
            return {'pred': paths}
            #output=feats.max(dim=-1)[1]
            

        if task=='NER':
            feats=F.relu(self.ner_linear(feats))
            logits = F.log_softmax(feats, dim=-1)
            paths, _ = self.ner_crf.viterbi_decode(logits, mask)
            return {'pred': paths}