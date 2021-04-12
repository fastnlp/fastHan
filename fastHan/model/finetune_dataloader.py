import numpy as np
from fastNLP import DataSet
from fastNLP.io import CTBLoader, CWSLoader, MsraNERLoader


def fastHan_CWS_Loader(url,chars_vocab,label_vocab):
    ds={'raw_words':[],'words':[],'target':[],'seq_len':[],'task_class':[]}
    #read file
    with open(url, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            if line:
                ds['raw_words'].append(line)

    #get chars and bmes
    for word_lst in ds['raw_words']:
        words=['[unused5]']
        target=['S']
        #word:str
        for word in word_lst:
            if len(word)==1:
                words.append(word)
                target.append('S')
            else:
                words=words+list(word)
                target.append('B')
                target=target+(len(word)-2)*['M']
                target.append('E')
        ds['words'].append(words)
        ds['target'].append(target)
        assert(len(words)==len(target))
        ds['seq_len'].append(len(words))
        ds['task_class'].append('CWS')
    
    del ds['raw_words']
    
    dataset=DataSet(ds)
    dataset.drop(lambda ins:ins['seq_len']>=500, inplace=True)
    chars_vocab.index_dataset(dataset,field_name='words',new_field_name='chars')
    label_vocab.index_dataset(dataset,field_name='target')
    
    dataset.set_input('chars','target','seq_len','task_class')
    dataset.set_target('target','seq_len')
    
    return dataset

def fastHan_NER_Loader(url,chars_vocab,label_vocab):

    def add_corpus(ins):
        ins['raw_chars'].insert(0,'[unused12]')
        return ins['raw_chars']
    def add_target(ins):
        ins['target'].insert(0,'O')
        return ins['target']

    dataset=MsraNERLoader()._load(url)

    dataset.apply(add_corpus,new_field_name='words')
    dataset.apply(add_target,new_field_name='target')
    dataset.delete_field('raw_chars')
    dataset.apply(lambda ins:len(ins['words']),new_field_name='seq_len')
    dataset.drop(lambda ins:ins['seq_len']>=500, inplace=True)
    dataset.apply(lambda ins:'NER',new_field_name='task_class')
    chars_vocab.index_dataset(dataset,field_name='words',new_field_name='chars')
    label_vocab.index_dataset(dataset,field_name='target')
    
    dataset.set_input('chars','target','seq_len','task_class')
    dataset.set_target('target','seq_len')
    
    return dataset

def fastHan_POS_loader(url,chars_vocab,label_vocab):
    def add_target(instance):
        pos=instance['pos']
        raw_words=instance['raw_words']
        target=['S-root']
        for i in range(len(pos)):
            l=len(raw_words[i])
            if l==1:
                target.append('S-'+pos[i])
            else:
                target.append('B-'+pos[i])
                for j in range(l-2):
                    target.append('M-'+pos[i])
                target.append('E-'+pos[i])
        return target

    def process(ins):
        s=''.join(ins['raw_words'])
        l=list(s)
        l.insert(0,'[unused14]')
        return l


    def process_dataset(dataset):
        dataset.delete_field('dep_head')
        dataset.delete_field('dep_label')
        dataset.apply(process,new_field_name='words')
        dataset.apply(add_target,new_field_name='target')
        dataset.delete_field('pos')
        dataset.delete_field('raw_words')
        dataset.apply(lambda x:len(x['words']),new_field_name='seq_len')
        for instance in dataset:
            if len(instance['target'])!=len(instance['words']):
                print('error',instance)
        return dataset

    def load_ctb(path):
        data=CTBLoader()._load(path)
        assert('pos' in data.field_arrays)
        assert('raw_words' in data.field_arrays)
        data=process_dataset(data)
        return data
    
    dataset=load_ctb(url)
    dataset.drop(lambda ins:ins['seq_len']>=500, inplace=True)
    dataset.apply(lambda x:'POS',new_field_name='task_class')
    chars_vocab.index_dataset(dataset,field_name='words',new_field_name='chars')
    label_vocab.index_dataset(dataset,field_name='target')
    
    return dataset

def fastHan_Parsing_Loader(url,chars_vocab,label_vocab):
    def process(ins):
        s=''.join(ins['raw_words'])
        l=list(s)
        return l

    # add seg_targets
    def add_segs(instance):
        words = instance['raw_words']
        segs = [0]*len(instance['words'])
        index = 0
        for word in words:
            index = index + len(word) - 1
            segs[index] = len(word)-1
            index = index + 1
        return segs

    # add target_masks
    def add_mask(instance):
        words = instance['raw_words']
        mask = []
        for word in words:
            mask.extend([0] * (len(word) - 1))
            mask.append(1)
        return mask


    def add_char_heads(instance):
        words = instance['raw_words']
        heads = instance['dep_head']
        char_heads = []
        char_index = 1  # 因此存在root节点所以需要从1开始
        head_end_indexes = np.cumsum(list(map(len, words))).tolist() + [0] # 因为root是0,0-1=-1
        for word, head in zip(words, heads):
            char_head = []
            if len(word)>1:
                char_head.append(char_index+1)
                char_index += 1
                for _ in range(len(word)-2):
                    char_index += 1
                    char_head.append(char_index)
            char_index += 1
            char_head.append(head_end_indexes[head-1])
            char_heads.extend(char_head)
        return char_heads

    def add_char_labels(instance):
        """
        将word_lst中的数据按照下面的方式设置label
        比如"复旦大学 位于 ", 对应的分词是"B M M E B E", 则对应的dependency是"复(dep)->旦(head)", "旦(dep)->大(head)"..
                对应的label是'app', 'app', 'app', , 而学的label就是复旦大学这个词的dependency label
        :param instance:
        :return:
        """
        words = instance['raw_words']
        labels = instance['dep_label']
        char_labels = []
        for word, label in zip(words, labels):
            for _ in range(len(word)-1):
                char_labels.append('APP')
            char_labels.append(label)
        return char_labels

    def process_dep_head(ins):
        l=[]
        for x in ins['dep_head']:
            l.append(int(x))
        return l

    def word_lens(ins):
        l=[0]
        length=0
        for word in ins['raw_words']:
            l.append(length+1)
            length+=len(word)
        return l

    def add_word_pairs(instance):
        # List[List[((head_start, head_end], (dep_start, dep_end]), ...]]
        word_end_indexes = np.array(list(map(len, instance['raw_words'])))
        word_end_indexes = np.cumsum(word_end_indexes).tolist()
        word_end_indexes.insert(0, 0)
        word_pairs = []
        pos_tags = instance['pos']
        for idx, head in enumerate(instance['dep_head']):
            if pos_tags[idx]=='PU': # 如果是标点符号，就不记录
                continue
            if head==0:
                word_pairs.append((('root', (word_end_indexes[idx], word_end_indexes[idx+1]))))
            else:
                word_pairs.append(((word_end_indexes[head-1], word_end_indexes[head]),
                                   (word_end_indexes[idx], word_end_indexes[idx + 1])))
        return word_pairs

    def add_label_word_pairs(instance):
        # List[List[((head_start, head_end], (dep_start, dep_end]), ...]]
        word_end_indexes = np.array(list(map(len, instance['raw_words'])))
        word_end_indexes = np.cumsum(word_end_indexes).tolist()
        word_end_indexes.insert(0, 0)
        word_pairs = []
        labels = instance['dep_label']
        pos_tags = instance['pos']
        for idx, head in enumerate(instance['dep_head']):
            if pos_tags[idx]=='PU': # 如果是标点符号，就不记录
                continue
            label = label_vocab['Parsing'].to_index(labels[idx])
            if head==0:
                word_pairs.append((('root', label, (word_end_indexes[idx], word_end_indexes[idx+1]))))
            else:
                word_pairs.append(((word_end_indexes[head-1], word_end_indexes[head]), label,
                                   (word_end_indexes[idx], word_end_indexes[idx + 1])))
        return word_pairs

    def add_target(instance):
        pos=instance['pos']
        raw_words=instance['raw_words']
        target=['S-root']
        for i in range(len(pos)):
            l=len(raw_words[i])
            if l==1:
                target.append('S-'+pos[i])
            else:
                target.append('B-'+pos[i])
                for j in range(l-2):
                    target.append('M-'+pos[i])
                target.append('E-'+pos[i])
        return target

    def add_pun_masks(instance):
        tags = instance['pos']
        pun_masks = []
        for word, tag in zip(instance['raw_words'], tags):
            if tag=='PU':
                pun_masks.extend([1]*len(word))
            else:
                pun_masks.extend([0]*len(word))
        return pun_masks

    def process_dataset(dataset):
        dataset.apply(process,new_field_name='words')
        dataset.apply(add_target,new_field_name='target')
        dataset.apply(process_dep_head,new_field_name='dep_head')
        dataset.apply(add_char_heads, new_field_name='char_heads')
        dataset.apply(add_char_labels, new_field_name='char_labels')
        dataset.apply(add_segs, new_field_name='seg_targets')
        dataset.apply(add_mask, new_field_name='seg_masks')
        dataset.apply(add_pun_masks,new_field_name='pun_masks')
        dataset.apply(add_word_pairs, new_field_name='gold_word_pairs', ignore_type=True)
        dataset.apply(add_label_word_pairs,new_field_name='gold_label_word_pairs', ignore_type=True)


        dataset.apply(lambda ins:[0]+ins['char_heads'],new_field_name='char_heads')
        dataset.apply(lambda ins:['APP']+ins['char_labels'],new_field_name='char_labels')
        dataset.apply(lambda ins:['[unused1]']+ins['words'],new_field_name='words')
        dataset.apply(lambda ins:['APP']+ins['dep_label'],new_field_name='dep_label')
        dataset.apply(lambda ins:['root']+ins['pos'],new_field_name='pos')
        dataset.apply(lambda ins:[0]+ins['dep_head'],new_field_name='dep_head')
        dataset.apply(lambda x:len(x['dep_head']),new_field_name='seq_len_for_wordlist')
        dataset.apply(lambda x:len(x['words']),new_field_name='seq_len')
        dataset.apply(word_lens,new_field_name='word_lens')
        return dataset


    def load_ctb(path):
        data=CTBLoader()._load(path)
        data=process_dataset(data)

        data.drop(lambda ins:ins['seq_len']>=500, inplace=True)

        chars_vocab.index_dataset(data,field_name='words',new_field_name='chars')
        data.apply(lambda ins:'Parsing',new_field_name='task_class')

        label_vocab['Parsing'].index_dataset(data,field_name='char_labels')
        label_vocab['Parsing'].index_dataset(data, field_name='dep_label')
        label_vocab['pos'].index_dataset(data,field_name='pos')
        label_vocab['POS'].index_dataset(data,field_name='target')

        data.set_input('seq_len_for_wordlist','target',
                                         'seq_len','chars','dep_head',
                                         'dep_label','pos','word_lens',
                                         'task_class')
        data.set_target('target','seq_len')
        data.set_target('seg_targets','seg_masks')
        data.set_target('gold_word_pairs','gold_label_word_pairs','pun_masks')
        #add
        data.set_pad_val('char_labels', -1)
        data.set_pad_val('char_heads', -1)
        return data
    
    dataset=load_ctb(url)
    return dataset
