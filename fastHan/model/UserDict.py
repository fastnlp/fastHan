#reference and modify from "https://blog.csdn.net/ANNILingMo/article/details/80879910"


class Trie:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}
        self.end = -1

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        curNode = self.root
        for c in word:
            if not c in curNode:
                curNode[c] = {}
            curNode = curNode[c]
        curNode[self.end] = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        curNode = self.root
        for c in word:
            if not c in curNode:
                return False
            curNode = curNode[c]
        # Doesn't end here
        if not self.end in curNode:
            return False
        return True

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        curNode = self.root
        for c in prefix:
            if not c in curNode:
                return False
            curNode = curNode[c]
        return True


class UserDict:
    def __init__(self):
        self.tree = Trie()
        self.tree_reverse = Trie()
        pass

    #根据词的列表构建trie树，包括正向和反向
    def load_list(self, word_list):
        for word in word_list:
            self.tree.insert(word)
            self.tree_reverse.insert(word[::-1])
        return 1

    #从文件中将词读取至一个list，并调用self.load_list
    def load_file(self, path, encoding='utf-8'):
        word_list = []
        with open(path, encoding=encoding) as file:
            for line in file:
                word = line.strip()
                word_list.append(word)
        return self.load_list(word_list)

    #根据输入的句子，进行正/反向最大匹配，返回匹配数最多的数量和结果（以BMESO标签的形式返回）
    def process_sentence(self, sentence, reverse=False):
        sentence = list(sentence)
        #正向匹配
        if reverse == False:
            tree = self.tree
        else:
            sentence = sentence[::-1]
            tree = self.tree_reverse

        sentence.append('<END-FASTHAN>')
        word_num = 0
        idx = 0
        tag_sequence = []
        word = ''
        #last_idx，上一个匹配到的词的末字符的索引
        last_idx = -1
        while idx < len(sentence):
            word = word + sentence[idx]
            if tree.search(word):
                last_idx = idx
            if not tree.startsWith(word) or idx == len(sentence) - 1:
                length_tag = len(tag_sequence)
                #如果匹配到，词的范围是[length_tag,last_idx]
                if length_tag == last_idx:
                    tag_sequence.append('s')
                    word_num += 1
                elif length_tag - 1 == last_idx:
                    tag_sequence.append('o')
                elif length_tag < last_idx:
                    word_num += 1
                    tag_sequence.append('b')
                    for i in range(length_tag + 1, last_idx):
                        tag_sequence.append('m')
                    tag_sequence.append('e')
                else:
                    raise ValueError('error when using dict')

                idx = len(tag_sequence)
                last_idx = idx - 1
                word = ''
            else:
                idx += 1

        assert (len(tag_sequence) == len(sentence))

        tag_sequence = tag_sequence[:-1]
        if reverse is False:
            return word_num, tag_sequence
        else:
            tag_sequence = tag_sequence[::-1]
            for i in range(len(tag_sequence)):
                if tag_sequence[i] == 'b':
                    tag_sequence[i] = 'e'
                elif tag_sequence[i] == 'e':
                    tag_sequence[i] = 'b'
            return word_num, tag_sequence

    def __call__(self, sentence):
        word_num, tag_sequence = self.process_sentence(sentence)
        word_num_reverse, tag_sequence_reverse = self.process_sentence(
            sentence, reverse=True)
        if word_num >= word_num_reverse:
            return word_num, tag_sequence
        return word_num_reverse, tag_sequence_reverse
