###
###
# To run this camr_to_tuples.py, you need following command with two arguments:
# ''' python camr_to_tuples.py -input camr.txt -output tuples.txt '''
# The input files are in camr text representation format, and output files are in camr tuples representation format.
###

# !/usr/bin/env python
# -*- coding:utf-8 -*-

from collections import defaultdict
import sys, re

# change this if needed
ERROR_LOG = sys.stderr

# change this if needed
DEBUG_LOG = sys.stderr


class CAMR(object):
    def __init__(self,
                 node_list=None,
                 node_value_list=None,
                 relation_list=None,
                 attribute_list=None):
        """
        node_list: names of nodes in CAMR graph, e.g. CAMR of "我 爱 你" has three nodes "x1", "x2" and "x3"
        node_value_list: values(concepts) of nodes in CAMR graph, e.g. concept "我" of node "x1"
        relation_list: list of relations and alignment of relations between two nodes
        attribute_list: list of attributes between node and its alignment, node "name" and tokens of named entity
        """
        # initialize CAMR graph nodes using list of nodes name
        # root, by default, is the first in node_list

        if node_list is None:
            self.nodes = []
            self.root = None
        else:
            self.nodes = node_list[:]
            if len(node_list) != 0:
                self.root = node_list[0]
            else:
                self.root = None
        if node_value_list is None:
            self.node_values = []
        else:
            self.node_values = node_value_list[:]
        if relation_list is None:
            self.relations = []
        else:
            self.relations = relation_list[:]
        if attribute_list is None:
            self.attributes = []
        else:
            self.attributes = attribute_list[:]

    @staticmethod
    def get_amr_line(input_f):
        """
        Read the file containing CAMRs. CAMRs are separated by a blank line.
        Each call of get_amr_line() returns the next available CAMR (in one-line form).
        """
        sid = ''
        id_token_dict = {}
        cur_amr = []
        has_content = False
        for line in input_f:
            line = line.strip()
            if '\ufeff' in line:
                line = line.replace('\ufeff', '')
            if '\u200b' in line:
                line = line.replace('\u200b', '')
            if line == "":
                if not has_content:
                    # empty lines before current CAMR
                    continue
                else:
                    # end of current CAMR
                    break
            if line.strip().startswith("#"):
                if '::id' in line:
                    sid = re.findall(r'# ::id export_amr\.(.*?)\s*::', line)[0]
                elif '::wid' in line:
                    wid = line[len('# ::wid '):].strip().split(' ')
                    for i in wid:
                        token_id, token = i.split('_')
                        if token != '':
                            # key: id number, value: token (e.g. "1":"我")
                            id_token_dict[int(token_id[1:])] = token
                else:
                    continue
            else:
                has_content = True
                if line[0] == ':' and '(' not in line.split()[0]:
                    temp = line.split()
                    temp[0] = temp[0] + '()'
                    line = ' '.join(temp)
                cur_amr.append(line.strip())
        return "".join(cur_amr), sid, id_token_dict

    @staticmethod
    def parse_AMR_line(line, align_dict):
        # Current state. It denotes the last significant symbol encountered. 1 for (, 2 for :, 3 for /,
        # and 0 for start state or ')'
        # Last significant symbol is ( --- start processing node name
        # Last significant symbol is : --- start processing relation name
        # Last significant symbol is / --- start processing node value (concept)
        # Last significant symbol is ) --- current node processing is complete
        # Note that if these symbols are inside parenthesis, they are not significant symbols.

        def update_triple(node_relation_dict, relation_align_dict, u, r, v):
            node_relation_dict[u].append((r, v))
            if u in relation_align_dict and relation_align_dict[u] != []:
                if relation_align_dict[u][-1][0] == r and isinstance(
                        relation_align_dict[u][-1][1], tuple) == 0:
                    old_value = relation_align_dict[u][-1][1]
                    new_value = (r, (old_value, v))
                    relation_align_dict[u].pop(-1)
                    relation_align_dict[u].append(new_value)

        def id_anchor(node_name):
            joint = node_name.split('_')
            joint_word = ''
            if node_name.count('x') == node_name.count('_') + 1:
                joint = node_name.split('x')
                for words in joint:
                    joint_word += words
            elif node_name.count('x') <= node_name.count('_'):
                joint_list = []
                if node_name.count('x') == 1:
                    for words in joint[1:]:
                        joint_word = joint[0][1:] + '.' + words
                        joint_list.append(joint_word)
                else:
                    joint = node_name.split('_x')
                    for mx in joint:
                        if mx[0] != 'x':
                            mx = 'x' + mx
                        if '_' in mx:
                            nj = mx.split('_')
                            for words in nj[1:]:
                                joint_word = nj[0][1:] + '.' + words
                                joint_list.append(joint_word)
                        else:
                            joint_list.append(mx[1:])
                joint_word = '_'.join(joint_list)
            return joint_word

        def anchor_token(anchor):
            try:
                int(anchor)
            except:
                if '_' not in anchor:
                    subword = anchor.split('.')
                    word = align_dict[int(subword[0])][int(subword[1]) - 1]
                else:
                    split_sequence = anchor.split('_')
                    joint_word = []
                    for w in split_sequence:
                        if '.' not in w:
                            joint_word.append(align_dict[int(w)])
                        else:
                            joint_word.append(align_dict[int(
                                w.split('.')[0])][int(w.split('.')[1]) - 1])
                    word = ''.join(joint_word)
            else:
                split_sequence = anchor.split('_')
                word = ''.join([align_dict[int(w)] for w in split_sequence])
            return word

        # start with '('
        state = 0
        # node stack
        stack = []
        # current not-yet-reduced character sequence
        cur_charseq = []
        # key: node value: concept
        node_dict = {}
        # node name list (order: occurrence of the node)
        node_name_list = []
        # named-entity list
        opx_list = []
        # key: word id on arc, value: function word
        arc_dict = {}
        # key: node name, value: list of (relation name, the other node name)
        node_relation_dict = defaultdict(list)
        # key: node name, value: list of (relation, (alignment, the other node name))
        relation_align_dict = defaultdict(list)
        # key: node name, value: list of ('anchor', id)
        node_align_dict = defaultdict(list)
        # key: node name, value: list of (relation, named entity)
        node_opx_dict = defaultdict(list)
        # key: node name, value: list of anaphor
        coref_dict = defaultdict(list)
        # current relation name
        cur_relation_name = ""
        # having unmatched quote string
        in_quote = False
        for i, c in enumerate(line.strip()):
            if c == " ":
                # allow space in relation name
                if state == 2:
                    cur_charseq.append(c)
                continue
            if c == "\"":
                # flip in_quote value when a quote symbol is encountered
                # insert placeholder if in_quote from last symbol
                in_quote = not in_quote
            elif c == "(":
                # not significant symbol if inside quote
                if in_quote:
                    cur_charseq.append(c)
                    continue
                # last symbol is ":", we get the relation name
                # e.g :arg0() ...
                # at this point we get "arg0"
                if state == 2:
                    # update current relation name for future use
                    cur_relation_name = "".join(cur_charseq).strip()
                    cur_charseq[:] = []
                    # e.g. :arg0(x2/的) (x3 / 我)
                    # prepare for relation alignment "x2/的"
                    relation_align_dict[cur_relation_name] = []
                # last symbol is ")"
                # e.g. arg0() (
                # at this point we clean "cur_charseq" ready for node
                elif state == 0:
                    cur_charseq[:] = []
                # current symbol is "("
                state = 1
            elif c == ":":
                # not significant symbol if inside quote
                if in_quote:
                    cur_charseq.append(c)
                    continue
                # last symbol is "/", now we encounter ":", so we get a concept
                # e.g. (x2 / 爱-01 :arg0()
                # at this point we get "爱-01"
                if state == 3:
                    node_value = "".join(cur_charseq)
                    # clear current char sequence
                    cur_charseq[:] = []
                    # pop node name ("x2" in the above example)
                    cur_node_name = stack[-1]
                    # if we get a node instead of a concept, there is pronominal anaphora
                    # e.g. (x22 / x6 :arg0()
                    # at this point we get "x6"
                    if node_value[0] == 'x':
                        coref_dict[node_value].append(cur_node_name)
                        anchor = id_anchor('x' + cur_node_name)
                        node_value = anchor_token(anchor)
                    if node_value[0] == "\"" and node_value[-1] == "\"":
                        node_value = node_value[1:-1]
                    # distinguish tokens of named entity and normal nodes
                    if opx_list == []:
                        node_dict[cur_node_name] = node_value
                    else:
                        # e.g. x1_x2 / name :op1 x1/北京 :op2 x2/天安门
                        # at this point we get "北京"
                        node_opx_dict[cur_node_name].append(
                            (opx_list[-1], node_value))
                        opx_list.pop()
                # last symbol is ")", now we encounter ":", so we get a node indicating argument sharing.
                # e.g. arg1() x3 :
                # at this point we get "x3"
                elif (state == 0) and (cur_charseq != []):
                    node_name = "".join(cur_charseq)
                    cur_charseq[:] = []
                    node_name = node_name.strip()[1:]
                    if cur_relation_name != "":
                        update_triple(node_relation_dict, relation_align_dict,
                                      stack[-1], cur_relation_name, node_name)
                elif state == 1:
                    print("Error in parsing AMR" + line[0:i + 1])
                    print(
                        "If there is a colon, slash or bracket in the concept, please put double quotation marks on the concept!"
                    )
                    return None, None
                state = 2
            elif c == "/":
                if in_quote:
                    cur_charseq.append(c)
                    continue
                # Last significant symbol is "(". Now we encounter "/"
                # we may get a node or a relation alignment
                # e.g. :arg0(x2/的) (x1 / 我
                # at this point we get "x1" or "x2"
                if state == 1:
                    node_name = "".join(cur_charseq)
                    cur_charseq[:] = []
                    # push the node name to stack
                    node_name = node_name.strip()[1:]
                    stack.append(node_name)
                    # e.g. (x2 / 爱-01 :arg0() (x1 / 我)
                    # cur_relation_name is arg1, node name is x1, upper node is x2
                    # so we have a relation arg0(x2, x1)
                    if cur_relation_name != "":
                        # judge whether there is a relation alignment
                        if cur_relation_name not in relation_align_dict:
                            update_triple(node_relation_dict,
                                          relation_align_dict, stack[-2],
                                          cur_relation_name, node_name)
                        else:
                            relation_align_dict[stack[-2]].append(
                                (cur_relation_name, node_name))
                            state = 3
                            continue
                    if node_name not in node_dict:
                        node_name_list.append(node_name)
                        # add it and anchor value to node_align_dict
                        if '_' not in node_name:
                            if int(node_name) <= max(align_dict.keys()):
                                node_align_dict[node_name].append(
                                    ('anchor', node_name))
                        else:
                            node_align_dict[node_name].append(
                                ('anchor', id_anchor('x' + node_name)))
                elif state == 2:
                    # e.g. :op1 x1/北京 :op2 x2/天安门
                    # at this point we get "op1", "op2"
                    opx = "".join(cur_charseq)
                    cur_charseq[:] = []
                    opx_list.append(opx.split()[0])
                else:
                    # error if in other state
                    print("Error in parsing AMR" + line[0:i + 1])
                    return None, None
                state = 3
            elif c == ")":
                if in_quote:
                    cur_charseq.append(c)
                    continue
                # stack should be non-empty to find upper level node
                if len(stack) == 0:
                    print("Unmatched parenthesis at position" + str(i) +
                          "in processing" + line[0:i + 1])
                    return None, None
                # e.g. :arg1() x3)
                if (state == 0) and (cur_charseq != []):
                    node_name = "".join(cur_charseq)
                    cur_charseq[:] = []
                    node_name = node_name.strip()[1:]
                    if cur_relation_name != "":
                        update_triple(node_relation_dict, relation_align_dict,
                                      stack[-1], cur_relation_name, node_name)
                # relation without alignment
                # e.g. :arg0()
                elif state == 1:
                    if cur_charseq == []:
                        del relation_align_dict[cur_relation_name]
                        state = 0
                        continue
                    else:
                        print("Error in parsing AMR" + line[0:i + 1])
                        return None, None
                # Last significant symbol is "/". Now we encounter ")"
                # e.g. :arg1() (x3 / 你)
                # we get "你" here
                elif state == 3:
                    node_value = "".join(cur_charseq)
                    cur_charseq[:] = []
                    cur_node_name = stack[-1]
                    if cur_relation_name in relation_align_dict:
                        # get relation alignment
                        arc_dict[cur_node_name] = node_value
                        del relation_align_dict[cur_relation_name]
                        stack.pop()
                        state = 0
                        continue
                    if node_value[0] == 'x':
                        coref_dict[node_value].append(cur_node_name)
                        anchor = id_anchor('x' + cur_node_name)
                        node_value = anchor_token(anchor)
                    if opx_list == []:
                        node_dict[cur_node_name] = node_value
                    else:
                        node_opx_dict[cur_node_name].append(
                            (opx_list[-1], node_value))
                        node_dict[cur_node_name] = ''.join([
                            each_name[1]
                            for each_name in node_opx_dict[cur_node_name]
                        ])
                        opx_list.pop()
                # pop from stack, as the current node has been processed
                stack.pop()
                state = 0
            else:
                # not significant symbols, so we just shift.
                cur_charseq.append(c)
        if relation_align_dict != {}:
            for k in list(relation_align_dict.keys()):
                if relation_align_dict[k] == []:
                    del relation_align_dict[k]
        # create data structures to initialize an CAMR
        node_value_list = []
        relation_list = []
        attribute_list = []
        for v in node_name_list:
            if v not in node_dict:
                print("Error: Node name not found" + v)
                return None, None
            else:
                node_value_list.append(node_dict[v])
            # build relation list and attribute list for this node
            node_rel_list = []
            node_attr_list = []
            if v in node_relation_dict:
                for v0 in node_relation_dict[v]:
                    if v0[1] not in node_dict:
                        print("Error: Node name not found" + v0[1])
                        return None, None
                    node_rel_list.append([v0[0], v0[1]])
            if v in relation_align_dict:
                for v2 in relation_align_dict[v]:
                    node_rel_list.append([(v2[1][0], arc_dict[v2[1][0]]),
                                          v2[1][1]])
            # each node has a relation list and attribute list
            relation_list.append(node_rel_list)
            attribute_list.append(node_attr_list)
        # add TOP
        node_name_list.insert(0, "0")
        node_value_list.insert(0, "root")
        relation_list.insert(0, [])
        relation_list[0].append(["top", node_name_list[1]])
        attribute_list.insert(0, [])
        result_amr = CAMR(node_name_list, node_value_list, relation_list,
                          attribute_list)
        return result_amr, coref_dict