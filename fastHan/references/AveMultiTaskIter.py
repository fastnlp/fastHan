import random
from numbers import Number

import numpy as np
import torch
from fastNLP import BatchIter, DataSet, DataSetIter, Instance, logger


def _to_tensor(batch, field_dtype):
    """

    :param batch: np.array()
    :param field_dtype: 数据类型
    :return: batch, flag. 如果传入的数据支持转为tensor，返回的batch就是tensor，且flag为True；如果传入的数据不支持转为tensor，
        返回的batch就是原来的数据，且flag为False
    """
    try:
        if field_dtype is not None and isinstance(field_dtype, type)\
                and issubclass(field_dtype, Number) \
                and not isinstance(batch, torch.Tensor):
            if issubclass(batch.dtype.type, np.floating):
                new_batch = torch.as_tensor(batch).float()  # 默认使用float32
            elif issubclass(batch.dtype.type, np.integer):
                new_batch = torch.as_tensor(batch).long()  # 复用内存地址，避免复制
            else:
                new_batch = torch.as_tensor(batch)
            return new_batch, True
        else:
            return batch, False
    except Exception as e:
        raise e


class MultiTaskIter(BatchIter):
    """
    DataSetIter 用于从 `DataSet` 中按一定的顺序, 依次按 ``batch_size`` 的大小将数据取出，
    组成 `x` 和 `y`::

        batch = DataSetIter(data_set, batch_size=16, sampler=SequentialSampler())
        num_batch = len(batch)
        for batch_x, batch_y in batch:
            # do stuff ...

    """
    def __init__(self, dataset_dict, batch_size=1, sampler=None, as_numpy=False,
                 num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, collate_fn=None):
        """
        
        :param dataset: :class:`~fastNLP.DataSet` 对象, 数据集
        :param int batch_size: 取出的batch大小
        :param sampler: 规定使用的 :class:`~fastNLP.Sampler` 方式. 若为 ``None`` , 使用 :class:`~fastNLP.SequentialSampler`.
    
            Default: ``None``
        :param bool as_numpy: 若为 ``True`` , 输出batch为 numpy.array. 否则为 :class:`torch.Tensor`.

            Default: ``False``
        :param int num_workers: 使用多少个进程来预处理数据
        :param bool pin_memory: 是否将产生的tensor使用pin memory, 可能会加快速度。
        :param bool drop_last: 如果最后一个batch没有batch_size这么多sample，就扔掉最后一个
        :param timeout: 生成一个batch的timeout值
        :param worker_init_fn: 在每个worker启动时调用该函数，会传入一个值，该值是worker的index。
        :param collate_fn: 用于将样本组合成batch的函数
        """
        data=dict()

        for task in dataset_dict:
            data[task]=dict()
            data[task]['dataset']=dataset_dict[task]
            data[task]['data_iter']=DataSetIter(dataset=data[task]['dataset'], batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,collate_fn=collate_fn)
            data[task]['iter']=iter(data[task]['data_iter'])
        
        self.data=data
        self.num_batches = 0
        for task in data.keys():
            self.num_batches+=data[task]['data_iter'].get_num_batches(len(data[task]['data_iter'].dataiter.sampler), batch_size, drop_last)
        self.batch_size = batch_size
        self.drop_last=drop_last
        self.cur_batch_indices = None

    def get_batch_indices(self):
        """
        获取当前已经输出的batch的index。

        :return:
        """
        return self.cur_batch_indices


    @property
    def dataset(self):
        d=DataSet()
        for key in self.data:
            for ins in self.data[key]['dataset']['chars']:
                ins=Instance(chars=ins)
                d.append(ins)

        return d

    def find_min_length(self):
        min_length=0
        for key in self.data.keys():
            if self.data[key]['num']==0:
                continue
            if min_length==0:
                min_length=self.data[key]['num']
            else:
                min_length=min(self.data[key]['num'],min_length)
        return min_length

    def random_choose(self):
        total=0
        for key in self.data.keys():
            total+=self.data[key]['num']
        randnum=random.randint(1,total)
        for key in self.data.keys():
            randnum-=self.data[key]['num']
            if randnum<=0:
                return key

    def __iter__(self):
        

        for task in self.data.keys():
            self.data[task]['iter']=iter(self.data[task]['data_iter'])
            self.data[task]['num']=self.data[task]['data_iter'].get_num_batches(len(self.data[task]['data_iter'].dataiter.sampler), self.batch_size, self.drop_last)

        while True:
            min_length=self.find_min_length()
            if min_length==0:
                break

            task=self.random_choose()
            batch=next(self.data[task]['iter'],None)
            batch_x, batch_y=batch
            yield batch_x,batch_y
            self.data[task]['num']-=1
