import re


def restore_camr(line, id_token_list):
    # 预处理内容
    line = re.sub('（', '(', line)
    line = re.sub('）', ')', line)
    line = re.sub('：', ':', line)
    time_list = re.findall('\d+\s*\:\s*\d+', line)
    line = re.sub('"', '', line)
    for s in time_list:
        ss = re.sub(':', '：', s)
        line = re.sub(s, ss, line)

    idx = 0
    amr_list = []

    new_mark = len(id_token_list) + 2
    node_dict = {}
    node_name_list = []

    def convert_node_value(node_value, is_bracket1=False):
        nonlocal id_token_list, node_name_list
        if not node_value:
            return node_value
        if '^' not in node_value:
            node_name = search_mark(node_value.split('-')[0])
            if not is_bracket1:
                node_name_list.append(node_name)
            return node_name + '/' + node_value
        else:
            node_value1, node_value2 = node_value.split('^')[0:2]
            node_name1 = search_mark(node_value1.split('-')[0])
            if int(node_name1[1:]) > len(id_token_list):
                node_name1 = 'x1'
            node_name2 = search_mark(node_value2.split('-')[0])
            if not node_name2 in node_name_list:
                node_name2 = node_name_list[0]
            return node_name1 + '/' + node_name2

    def search_mark(node_value):
        nonlocal id_token_list, node_dict, new_mark
        node_name = node_dict.get(node_value, 0)
        if node_name > len(id_token_list):
            node_name = new_mark
            new_mark += 1
            node_dict[node_value] = node_name
            return 'x' + str(node_name)

        i = node_name + 1
        while i <= len(id_token_list):
            if id_token_list[i] == node_value:
                break
            i += 1
        if i > len(id_token_list):
            j = 1
            while j <= node_name:
                if id_token_list[j] == node_value:
                    break
                j += 1
            if node_name >= 1 and j <= node_name:
                node_name = j
        else:
            node_name = i

        if node_name <= 0:
            node_name = new_mark
            new_mark += 1

        node_dict[node_value] = node_name
        return 'x' + str(node_name)

    # 递归处理部分
    def bracket1():
        nonlocal idx
        nonlocal amr_list
        cur_charseq = []
        cur_idx = idx
        has_content = False
        while True:
            if cur_idx >= len(line):
                break
            if line[cur_idx] == '(' and not has_content:
                has_content = True
            if has_content:
                if line[cur_idx] == ':':
                    cur_idx = idx
                    cur_charseq = []
                    break
                if line[cur_idx] == '(' and cur_charseq:
                    break
                cur_charseq.append(line[cur_idx])
                if cur_charseq[-1] == ')':
                    has_content = False
                    cur_idx += 1
                    break
            cur_idx += 1

        idx = cur_idx
        if not cur_charseq:
            amr_list.append('(')
            amr_list.append(')')
            cur_node_value = ''
        elif cur_charseq[-1] != ')':
            cur_node_value = get_seq_value(cur_charseq[1:])
            amr_list.append('(')
            amr_list.append(convert_node_value(cur_node_value, True))
            amr_list.append(')')
        else:
            cur_node_value = get_seq_value(cur_charseq[1:-1])
            amr_list.append('(')
            amr_list.append(convert_node_value(cur_node_value, True))
            amr_list.append(')')
        pass

    def bracket2():
        nonlocal idx
        nonlocal amr_list
        nonlocal id_token_list
        cur_charseq = []
        cur_idx = idx

        # 寻找左括号
        while cur_idx < len(line):
            if line[cur_idx] == '(':
                break
            cur_idx += 1
        if cur_idx < len(line):
            amr_list.append('(')
        else:
            amr_list.append('(')
            amr_list.append(convert_node_value('-'))
            amr_list.append(')')
            idx = cur_idx
            return

        # 寻找变量名
        cur_idx += 1
        while cur_idx < len(line):
            if line[cur_idx] == ')' or line[cur_idx] == ':':
                break
            if line[cur_idx] == '(':
                break
            cur_charseq.append(line[cur_idx])
            cur_idx += 1
        cur_node_value = get_seq_value(cur_charseq)
        if not cur_node_value:
            cur_node_value = '-'
        amr_list.append(convert_node_value(cur_node_value))
        cur_charseq.clear()

        if cur_node_value == 'name':
            name_idx = len(amr_list) - 1
            n_value_list = []
            while cur_idx < len(line) and line[cur_idx] != ')':
                cur_charseq.append(line[cur_idx])
                cur_idx += 1
            cur_charseq.append(':')
            l = r = 0
            r_value = ''
            for i, ch in enumerate(cur_charseq):
                if ch == ':':
                    l = i
                    n_value = get_seq_value(cur_charseq[r:l])
                    if not n_value:
                        n_value = '-'
                    if r_value:
                        amr_list.append(' ')
                        amr_list.append(r_value)
                        amr_list.append(' ')
                        n_value = convert_node_value(n_value, True)
                        if n_value.split('/')[-1].startswith('x'):
                            n_value = n_value.split(
                                '/')[0] + '/' + id_token_list[int(
                                    n_value.split('/')[0][1:])]
                        amr_list.append(n_value)
                        n_value_list.append(n_value.split('/')[0])
                        n_value = r_value = ""
                elif ch.isspace() and i > 0 and cur_charseq[i - 1].isdigit():
                    r = i
                    r_value = get_seq_value(cur_charseq[l:r])
                    if not r_value:
                        r_value = 'op1'
                    r_value = ':' + r_value
            cur_idx += 1
            amr_list.append(')')
            idx = cur_idx
            if n_value_list:
                amr_list[name_idx] = '_'.join(n_value_list) + '/' + 'name'
            return

        # 寻找关系名
        def relation():
            nonlocal cur_idx, idx
            cur_relationseq = []
            relation_content = False
            while cur_idx < len(line):
                if line[cur_idx] == ')':
                    return relation_content
                if line[cur_idx] == ':':
                    break
                cur_idx += 1

            if cur_idx >= len(line):
                return relation_content

            while cur_idx < len(line):
                if line[cur_idx] == ':':
                    relation_content = True
                    cur_relationseq = []
                if relation_content:
                    if line[cur_idx] == '(':
                        cur_relation_value = get_seq_value(cur_relationseq)
                        amr_list.append(' ')
                        amr_list.append(':' + cur_relation_value)
                        idx = cur_idx
                        bracket1()
                        bracket2()
                        cur_idx = idx
                        break
                    elif line[cur_idx] == ')':
                        cur_relation_value = get_seq_value(cur_relationseq)
                        amr_list.append(' ')
                        amr_list.append(':' + cur_relation_value)
                        amr_list.append('()(')
                        amr_list.append(convert_node_value('-'))
                        amr_list.append(')')
                        cur_idx += 1
                        break
                    cur_relationseq.append(line[cur_idx])
                cur_idx += 1
            idx = cur_idx
            return relation_content

        while True:
            if not relation():
                break

        while cur_idx < len(line):
            if line[cur_idx] == ')':
                break
            cur_idx += 1
        amr_list.append(')')
        cur_idx = cur_idx + 1
        idx = cur_idx
        return

    bracket2()
    return amr_list


def get_seq_value(cur_charseq):
    for idx, ch in enumerate(cur_charseq):
        if ch.isspace():
            cur_charseq[idx] = ''
    s = ''.join(cur_charseq)
    s = re.sub(':', '', s)
    s = re.sub('\(', '', s)
    s = re.sub('\)', '', s)
    return s


def convert_camr_to_lines(amr):
    amr_list = amr.split(':')
    for i, line in enumerate(amr_list):
        if i == 0:
            continue
        amr_list[i] = ':' + line

    num = 0
    for i, line in enumerate(amr_list):
        if '(' not in line:
            continue
        amr_list[i] = '\t' * num + line
        num = num + line.count('(') - line.count(')')

    for i, line in enumerate(amr_list):
        if i == 0:
            continue
        if '(' in line:
            amr_list[i] = '\n' + line

    return ''.join(amr_list) + '\n'