from fastNLP.core.metrics import Metric
from fastNLP import SpanFPreRecMetric
import torch

from .metrics import SegAppCharParseF1Metric,CWSMetric

class MultiTaskMetric(Metric):
    def __init__(self,all_tasks,task_vocab_map,biaffine_task='Parsing-ctb9'):
        super().__init__()
        self.all_tasks=all_tasks
        self.task_vocab_map=task_vocab_map
        self.biaffine_task=biaffine_task
        self.metrics=dict()
        
        for task in all_tasks:
            if task==biaffine_task:
                self.metrics[task]=SegAppCharParseF1Metric(pun_index=task_vocab_map[self.biaffine_task].word2idx['punct'])
                continue    
            self.metrics[task]=SpanFPreRecMetric(tag_vocab=task_vocab_map[task])
        
        self.parsing_cws_metric=CWSMetric()
        
        self.tasks_flag=set()
    
    def update(self,task,seq_len,labels,pred=None,heads=None,head_preds=None,label_preds=None):
        task=task.tolist()[0]
        task=self.all_tasks[task]
        self.tasks_flag.add(task)
        if task==self.biaffine_task:
            assert heads is not None
            assert head_preds is not None
            assert label_preds is not None
            self.metrics[task].update(labels,heads,head_preds,label_preds,seq_len)
            self.parsing_cws_metric.update(labels,label_preds,seq_len)
        else:
            assert pred is not None
            self.metrics[task].update(pred=pred,target=labels[:,1:],seq_len=seq_len)
    
    def reset(self):
        for task in self.metrics:
            self.metrics[task].reset()
    
    def get_metric(self,reset=True):
        scores=dict()
        for task in self.tasks_flag:
            macro_task,corpus=task.split('-')
            if macro_task not in scores:
                scores[macro_task]=dict()
            scores[macro_task][corpus]=self.metrics[task].get_metric()
        
        
        all_f=[]
        for macro_task in scores:
            ave_f=sum(map(lambda corpus:scores[macro_task][corpus]['f'],scores[macro_task]))/len(scores[macro_task])
            all_f.append(ave_f)

        scores['avg_f']=sum(all_f)/len(all_f)
        if self.biaffine_task in self.tasks_flag:
            scores['Parsing']['ctb9-cws']=self.parsing_cws_metric.get_metric(reset=reset)
        
        scores['all_f']=all_f
        if reset:
            self.tasks_flag=set()
        return scores