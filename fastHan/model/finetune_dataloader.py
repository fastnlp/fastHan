import itertools
import numpy as np

from fastNLP import DataSet

MAX_LEN = 300


def fastHan_CWS_Loader(lines, label_to_index, tokenizer):
    data = {
        'input_ids': [],
        'attention_mask': [],
        'labels': [],
        'seq_len': [],
    }

    for line in lines:
        line = line.strip()
        if len(line) <= 1:
            continue

        line = line.split(' ')

        words = []
        labels = []
        for word in line:
            if len(word) == 0:
                continue
            words = words + list(word)
            if len(word) == 1:
                labels.append('s')
            else:
                labels = labels + ['b'] + ['m'] * (len(word) - 2) + ['e']

        words = words[:MAX_LEN]
        labels = labels[:MAX_LEN]
        labels = [-100] + [label_to_index[x] for x in labels] + [-100]

        tokenize_result = tokenizer(words, is_split_into_words=True)

        if len(tokenize_result['input_ids']) != len(labels):
            continue

        data['input_ids'].append(tokenize_result['input_ids'])
        data['attention_mask'].append(tokenize_result['attention_mask'])
        data['labels'].append(labels)
        data['seq_len'].append(len(labels) - 2)

    return DataSet(data)


def fastHan_POS_loader(lines, label_to_index, tokenizer):
    data = {
        'input_ids': [],
        'attention_mask': [],
        'labels': [],
        'seq_len': [],
    }

    words = []
    labels = []

    for line in lines:
        if line == '\n' and len(words) > 0:
            words = words[:MAX_LEN]
            labels = labels[:MAX_LEN]
            labels = [-100] + [label_to_index[x] for x in labels] + [-100]

            tokenize_result = tokenizer(words, is_split_into_words=True)
            if len(tokenize_result['input_ids']) == len(labels):
                data['input_ids'].append(tokenize_result['input_ids'])
                data['attention_mask'].append(
                    tokenize_result['attention_mask'])
                data['labels'].append(labels)
                data['seq_len'].append(len(labels) - 2)
            words = []
            labels = []
        else:
            line = line.strip()
            line = line.split('\t')
            word = line[1]
            label = line[3].lower()

            words = words + list(word)
            if len(word) == 1:
                labels.append('s-' + label)
            else:
                labels = labels + [
                    'b-' + label
                ] + ['m-' + label] * (len(word) - 2) + ['e-' + label]
    return DataSet(data)


# def fastHan_CWS_guwen_Loader():
#     pass


def fastHan_POS_guwen_loader(lines, label_to_index, tokenizer):
    data = {'input_ids': [], 'attention_mask': [], 'labels': [], 'seq_len': []}

    for line in lines:
        line = line.strip()
        if len(line) <= 1:
            continue

        line = line.split(' ')

        words = []
        labels = []
        for word_label in line:
            if len(word_label) == 0:
                continue

            word, label = word_label.split('/')
            words = words + list(word)
            if len(word) == 1:
                labels.append('s-' + label)
            else:
                labels = labels + [
                    'b-' + label
                ] + ['m-' + label] * (len(word) - 2) + ['e-' + label]

        words = words[:MAX_LEN]
        labels = labels[:MAX_LEN]
        labels = [-100] + [label_to_index[x] for x in labels] + [-100]

        tokenize_result = tokenizer(words, is_split_into_words=True)

        if len(tokenize_result['input_ids']) != len(labels):
            continue

        data['input_ids'].append(tokenize_result['input_ids'])
        data['attention_mask'].append(tokenize_result['attention_mask'])
        data['labels'].append(labels)
        data['seq_len'].append(len(labels) - 2)

    return DataSet(data)


def fastHan_NER_Loader(lines, label_to_index, tokenizer):
    data = {
        'input_ids': [],
        'attention_mask': [],
        'labels': [],
        'seq_len': [],
    }

    words = []
    labels = []

    for line in lines:
        if line == '\n' and len(words) > 0:
            words = words[:MAX_LEN]
            labels = labels[:MAX_LEN]
            labels = [-100] + [label_to_index[x] for x in labels] + [-100]

            tokenize_result = tokenizer(words, is_split_into_words=True)
            if len(tokenize_result['input_ids']) == len(labels):
                data['input_ids'].append(tokenize_result['input_ids'])
                data['attention_mask'].append(
                    tokenize_result['attention_mask'])
                data['labels'].append(labels)
                data['seq_len'].append(len(labels) - 2)
            words = []
            labels = []
        else:
            line = line.strip()
            word, label = line.split(' ')
            label = label.lower()

            words.append(word)
            labels.append(label)

    return DataSet(data)


def fastHan_Parsing_Loader(lines, label_to_index, tokenizer):
    data = {
        'input_ids': [],
        'attention_mask': [],
        'heads': [],
        'labels': [],
        'seq_len': []
    }

    words = []
    heads = []
    labels = []

    skip_1 = 0
    skip_2 = 0
    for line in lines:
        if line == '\n' and len(words) > 0:
            char_words = list(itertools.chain(*words))
            if len(char_words) > 300:
                skip_1 += 1
                words = []
                heads = []
                labels = []
                continue
            tokenize_result = tokenizer(char_words, is_split_into_words=True)
            if len(tokenize_result['input_ids']) - 2 != len(char_words):
                skip_2 += 1
                words = []
                heads = []
                labels = []
                continue

            # 添加根结点
            tokenize_result['input_ids'].insert(1, 1)
            tokenize_result['attention_mask'].insert(1, 1)

            head_end_indexes = np.cumsum(list(map(len, words))).tolist() + [0]
            char_index = 1

            char_heads = []
            char_labels = []

            for word, head, label in zip(words, heads, labels):
                for _ in range(len(word) - 1):
                    char_index += 1
                    char_heads.append(char_index)
                    char_labels.append('app')
                char_index += 1
                char_heads.append(head_end_indexes[head - 1])
                char_labels.append(label)

            # 根节点的label都是-100
            labels = [-100] + [label_to_index[x] for x in char_labels]
            char_heads = [-100] + char_heads

            data['input_ids'].append(tokenize_result['input_ids'])
            data['attention_mask'].append(tokenize_result['attention_mask'])
            data['labels'].append(labels)
            data['heads'].append(char_heads)
            data['seq_len'].append(len(labels))

            words = []
            heads = []
            labels = []
        else:
            line = line.strip()
            line = line.split('\t')

            word = line[1]
            head = line[6].lower()
            label = line[7].lower()

            words.append(word)
            labels.append(label)
            heads.append(int(head))
    return DataSet(data)