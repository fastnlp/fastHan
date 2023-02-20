import re
import torch
import pandas as pd

from datasets import Dataset
from .camr_to_tuples import CAMR

MAX_LEN = 300


def FastCAMR_Parsing_Loader(data_path, tokenizer):
    sid_list, sent_list, id_token_list, amr_list, convert_amr_list = var_free_camrs(
        data_path)
    input_ids, attention_mask, decoder_input_ids, labels = [], [], [], []
    discard_num = 0
    discard_index = []
    for idx, sid in enumerate(sid_list):
        tokenize_result = tokenizer(convert_amr_list[idx])
        if len(tokenize_result['input_ids']) > MAX_LEN:
            discard_num += 1
            discard_index.append(idx)
            continue
        sent_tokenize_result = tokenizer(' '.join(sent_list[idx]),
                                         max_length=MAX_LEN,
                                         padding='max_length',
                                         truncation=True)
        amr_tokenize_result = tokenizer(convert_amr_list[idx],
                                        max_length=MAX_LEN,
                                        padding='max_length',
                                        truncation=True)
        input_ids.append(sent_tokenize_result['input_ids'])
        attention_mask.append(sent_tokenize_result['attention_mask'])
        decode_ids = amr_tokenize_result['input_ids']
        decoder_input_ids.append(decode_ids[:-1])
        labels.append(decode_ids[1:])
    # print("******* {0} sentences are discard! *******".format(discard_num))
    for idx in reversed(discard_index):
        sid_list.pop(idx)
        sent_list.pop(idx)
        id_token_list.pop(idx)
        amr_list.pop(idx)
        convert_amr_list.pop(idx)
    amr_data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels
    }
    amr_data = pd.DataFrame(amr_data)
    dataset = Dataset.from_pandas(amr_data, preserve_index=False)
    return dataset


def data_collator(amr_data):
    first = amr_data[0]
    batch = {}
    for k, v in first.items():
        if isinstance(v, torch.Tensor):
            batch[k] = torch.stack([f[k] for f in amr_data])
        else:
            batch[k] = torch.tensor([f[k] for f in amr_data])
    return batch


def var_free_camrs(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sid_list, sent_list, id_token_list, amr_list = read_raw_camrs(lines)
    convert_amr_list = [
        delete_camr_variables(convert_camr_to_single_line(amr),
                              id_token_list[idx])
        for idx, amr in enumerate(amr_list)
    ]
    return sid_list, sent_list, id_token_list, amr_list, convert_amr_list


def read_raw_camrs(lines):
    sid_list, sent_list, id_token_list, amr_list = [], [], [], []
    # 迭代输入文件中的每个句子
    cur_sent, cur_amr = [], []
    id_token_dict = {}
    has_content = False
    for line in lines:
        line = line.strip()
        if '\ufeff' in line:
            line = line.replace('\ufeff', '')
        if '\u200b' in line:
            line = line.replace('\u200b', '')
        if line == "":
            if has_content:  # end of current CAMR
                sent_list.append(cur_sent)
                id_token_list.append(id_token_dict)
                amr_list.append(cur_amr)
                cur_sent, cur_amr = [], []
                id_token_dict = {}
                has_content = False
            continue
        if line.strip().startswith("#"):
            if '::id' in line:
                sid = re.findall(r'# ::id export_amr\.(.*?)\s*::', line)[0]
                sid_list.append(sid)
            elif '::wid' in line:
                wid = line[len('# ::wid '):].strip().split(' ')
                for i in wid:
                    token_id, token = i.split('_')
                    if token != '':
                        # key: id number, value: token (e.g. "1":"我")
                        cur_sent.append(token)
                        id_token_dict[int(token_id[1:])] = token
            else:
                continue
        else:
            has_content = True
            cur_amr.append(line)
    if has_content:
        sent_list.append(cur_sent)
        id_token_list.append(id_token_dict)
        amr_list.append(cur_amr)
    return sid_list, sent_list, id_token_list, amr_list


def convert_camr_to_single_line(amr):
    return "".join([line.strip() for line in amr])


def delete_camr_variables(amr, id_token_dict):
    result_amr, coref_dict = CAMR.parse_AMR_line(amr, id_token_dict)
    node_dict = dict(zip(result_amr.nodes, result_amr.node_values))
    var = 'x\d+(?:_\d+)*(?:_x\d+(?:_\d+)*)*'
    coref_vars = '(' + var + '\s*/\s*' + var + ')'
    coref_var_list = re.findall(coref_vars, amr)
    for coref_v in coref_var_list:
        var0 = coref_v.split('/')[0].strip()
        var1 = coref_v.split('/')[1].strip()
        var0 = node_dict[var0[1:]]
        var1 = node_dict[var1[1:]]
        amr = re.sub(coref_v, var0 + '^' + var1, amr)
    normal_var = var + '\s*/\s*'
    amr = re.sub(normal_var, '', amr)
    return amr