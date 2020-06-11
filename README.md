# fastHan
## 简介
fastHan是基于[fastNLP](https://github.com/fastnlp/fastNLP)与pytorch实现的中文自然语言处理工具，像spacy一样调用方便。

其内核为基于BERT的联合模型，其在13个语料库中进行训练，可处理中文分词、词性标注、依存分析、命名实体识别四项任务。fastHan共有base与large两个版本，分别利用BERT的前四层与前八层。base版本在总参数量150MB的情况下各项任务均有不错表现，large版本则接近甚至超越SOTA模型。


## 安装指南
fastHan需要以下依赖的包：

torch>=1.0.0

fastNLP>=0.5.0

可执行如下命令完成安装：

```
pip install fastHan
```

## 使用教程
fastHan的使用极为简单，只需两步：加载模型、将句子输入模型。
#### 加载模型
执行以下代码可以加载模型：

```
from fastHan import FastHan
model=FastHan()
```
此时若用户为首次初始化模型，将自动从服务器中下载参数。

模型默认初始化为base，如果使用large版本，可在初始化时加入如下参数：

```
model=FastHan(model_type="large")
```
#### 输入句子
模型对句子进行依存分析的简单例子如下：

```
sentence="郭靖是金庸笔下的一名男主。"
answer=model(sentence,target="Parsing")
print(answer)
answer=model(sentence,target="NER")
print(answer)
```
模型将会输出如下信息：

```
[[['郭靖', 2, 'top', 'NR'], ['是', 0, 'root', 'VC'], ['金庸', 4, 'nn', 'NR'], ['笔', 5, 'lobj', 'NN'], ['下', 10, 'assmod', 'LC'], ['的', 5, 'assm', 'DEG'], ['一', 8, 'nummod', 'CD'], ['名', 10, 'clf', 'M'], ['男', 10, 'amod', 'JJ'], ['主', 2, 'attr', 'NN'], ['。', 2, 'punct', 'PU']]]
[[['郭靖', 'NR'], ['金庸', 'NR']]]
```
**任务选择**

target参数可在'Parsing'、'CWS'、'POS'、'NER'四个选项中取值，模型将分别进行依存分析、分词、词性标注、命名实体识别任务,模型默认进行CWS任务。其中词性标注任务包含了分词的信息，而依存分析任务又包含了词性标注任务的信息。命名实体识别任务相较其他任务独立。

如果分别运行CWS、POS、Parsing任务，模型输出的分词结果等可能存在冲突。如果想获得不冲突的各类信息，请直接运行包含全部所需信息的那项任务。

模型的POS、Parsing任务均使用CTB数据集进行训练，使用CTB标签集。NER使用msra标签集。

**分词风格**

模型是在13个语料库中进行训练，其中包含了10个分词语料库。不同语料库的分词粒度均不同，如本模型默认的CTB语料库分词粒度较细。如果想切换不同的粒度，可以使用模型的set_cws_style函数，例子如下：
>
```
sentence="一个苹果。"
print(model(sentence,'CWS'))
model.set_cws_style('cnc')
print(model(sentence,'CWS'))
```
模型将输出如下内容：

```
[['一', '个', '苹果', '。']]
[['一个', '苹果', '。']]
```
**输入与输出**

输入模型的可以是单独的字符串，也可是由字符串组成的列表。如果输入的是列表，模型将一次性处理所有输入的字符串，所以请自行控制 batch size。

模型的输出是在fastHan模块中定义的sentence与token类。模型将输出一个由sentence组成的列表，而每个sentence又由token组成。每个token本身代表一个被分好的次，有pos、head、head_label、ner四项属性，代表了该词的词性、依存关系、命名实体识别信息。

一则输入输出的例子如下所示：

```
sentence=["我爱踢足球。","林丹是冠军"]
answer=model(sentence,'Parsing')
for i,sentence in enumerate(answer):
    print(i)
    for token in sentence:
        print(token,token.pos,token.head,token.head_label)
```
模型将输入如下内容：

```
0
我 PN 2 nsubj
爱 VV 0 root
踢 VV 2 ccomp
足球 NN 3 dobj
。 PU 2 punct
1
林丹 NR 2 top
是 VC 0 root
冠军 NN 2 attr
！ PU 2 punct
```
可在分词风格中选择'as'、'cityu'进行繁体字分词，这两项为繁体语料库。

此外，由于各项任务共享词表、词嵌入，及时不切换模型的分词风格，模型对繁体字、英文字母、数字均具有一定识别能力。

**切换设备**

可使用模型的set_device函数，令模型在cuda上运行或切换回cpu，示例如下：

```
model.set_device('cuda:0')
model.set_device('cpu')
```
## 模型表现

模型在以下数据集进行测试和训练：

- CWS：AS, CITYU, CNC, CTB, MSR, PKU, SXU, UDC, WTB, ZX
- NER：MSRA、OntoNotes
- POS & Parsing：CTB9

最终模型在各项任务中取得的F值如下：


任务 | CWS | Parsing | POS | NER MSRA | NER OntoNotes
---|---|--- |--- |--- |---
base模型 | 97.38 | 81.22,76.71 | 95.19 | 94.33 | 82.86
large模型 | 97.52 | 85.52,81.38 | 95.92 | 95.50 | 83.82

表格中单位为百分数。CWS的成绩是10项任务的平均成绩。Parsing中的两个成绩分别代表Fudep和Fldep。

目前此工作准备投稿，如被录用将会把链接附于此处，届时可在论文中得到更多关于模型结构及训练的详细信息。