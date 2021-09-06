# fastHan

For English README, you can click [here](https://github.com/fastnlp/fastHan/blob/master/README.EN.md)


## 简介
fastHan是基于[fastNLP](https://github.com/fastnlp/fastNLP)与pytorch实现的中文自然语言处理工具，像spacy一样调用方便。

其内核为基于BERT的联合模型，其在13个语料库中进行训练，可处理中文分词、词性标注、依存分析、命名实体识别四项任务。fastHan共有base与large两个版本，分别利用BERT的前四层与前八层。base版本在总参数量150MB的情况下各项任务均有不错表现，large版本则接近甚至超越SOTA模型。

## 引用

如果您在工作中使用了fastHan工具，您可以引用这篇[论文](https://arxiv.org/abs/2009.08633)：
Zhichao Geng, Hang Yan, Xipeng Qiu and Xuanjing Huang, fastHan: A BERT-based Multi-Task Toolkit for Chinese NLP, ACL, 2021.
```
@inproceedings{geng-etal-2021-fasthan,
  author = {Geng, Zhichao and Yan, Hang and Qiu, Xipeng and Huang, Xuanjing},
  title = {fastHan: A BERT-based Multi-Task Toolkit for Chinese NLP},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: System Demonstrations},
  year = {2021},
  pages = {99--106}, 
  url = {https://aclanthology.org/2021.acl-demo.12}
}
```


## 安装指南
fastHan需要以下依赖的包：

- torch>=1.0.0
- fastNLP>=0.5.5

**版本更新:**
- 1.1版本的fastHan与0.5.5版本的fastNLP会导致importerror。如果使用1.1版本的fastHan，请使用0.5.0版本的fastNLP。
- 1.2版本的fastHan修复了fastNLP版本兼容问题。小于等于1.2版本的fastHan在输入句子的首尾包含**空格、换行**符时会产生BUG。如果字符串首尾包含上述字符，请使用 strip 函数处理输入字符串。
- 1.3版本的fastHan自动对输入字符串做 strip 函数处理。
- 1.4版本的fastHan加入用户词典功能（仅限于分词任务）
- 1.5版本的fastHan
  - 修正了Parsing任务中可能会出现的ValueError
  - 修改结果的返回形式，默认以list的形式返回
  - 可以通过url路径加载模型
- 1.6版本的fastHan
  - 将用户词典功能扩充到所有任务
  - 可以在返回值中包含位置信息
- 1.7版本的fastHan
  - 添加finetune功能

可执行如下命令完成安装：

```
pip install fastHan
```

或者可以通过github安装：
```
git clone git@github.com:fastnlp/fastHan.git
cd fastHan
python setup.py install
```

## **快速上手**
fastHan的使用极为简单，只需两步：加载模型、将句子输入模型。

**加载模型**

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

此外，对于手动下载模型的用户以及保存过微调模型的用户，可以使用模型路径加载模型。下载压缩包并解压后，可将对应路径通过url参数传入。注意，此处也要根据使用的模型传入正确的model_type参数。一则使用模型路径加载模型的例子如下：
```
model=FastHan(model_type='large',url="C:/Users/gzc/.fastNLP/fasthan/fasthan_large")
```

 **输入句子**

模型对句子进行依存分析、命名实体识别的简单例子如下：

```
sentence="郭靖是金庸笔下的男主角。"
answer=model(sentence)
print(answer)
answer=model(sentence,target="Parsing")
print(answer)
answer=model(sentence,target="NER")
print(answer)
```
模型将会输出如下信息：

```
[['郭靖', '是', '金庸', '笔', '下', '的', '男', '主角', '。']]
[[['郭靖', 2, 'top', 'NR'], ['是', 0, 'root', 'VC'], ['金庸', 4, 'nn', 'NR'], ['笔', 5, 'lobj', 'NN'], ['下', 8, 'assmod', 'LC'], ['的', 5, 'assm', 'DEG'], ['男', 8, 'amod', 'JJ'], ['主角', 2, 'attr', 'NN'], ['。', 2, 'punct', 'PU']]]
[[['郭靖', 'NR'], ['金庸', 'NR']]]
```
可选参数：
- **target**: 可在'Parsing'、'CWS'、'POS'、'NER'四个选项中取值，模型将分别进行依存分析、分词、词性标注、命名实体识别任务,模型默认进行CWS任务。
  - 词性标注任务包含了分词的信息，而依存分析任务又包含了词性标注任务的信息。命名实体识别任务相较其他任务独立。
  - 模型的POS、Parsing任务均使用CTB标签集。NER使用msra标签集。
- **use_dict**: 是否使用用户词典，默认为False。
- **return_list**：是否以list形式传递返回值。默认为True。
- **return_loc**: 是否将词的位置信息返回，默认为False。可用于spanF metric的评估。

**切换设备**

可使用模型的 set_device 函数，令模型在cuda上运行或切换回cpu，示例如下：

```
model.set_device('cuda:0')
model.set_device('cpu')
```
## **进阶功能**

**微调模型**

用户可以根据自己的需求在新的数据集上进行微调，一则微调的例子如下方所示：
```
from fastHan import FastHan

model=FastHan('large')

# traindata file path
cws_url='train.dat'

model.set_device(0)
model.finetune(data_path=cws_url,task='CWS',save=True,save_url='finetuned_model')
```
微调前设置set_device函数可实用GPU加速。微调时需要将用于训练的数据按格式放到一个文件里。

对于CWS任务，则要求每行一条数据，每个词用空格分隔开。

Example:
 
    上海 浦东 开发 与 法制 建设 同步
    新华社 上海 二月 十日 电 （ 记者 谢金虎 、 张持坚 ）
    ...

对于NER任务，要求按照MSRA数据集的格式与标签集。

Example:

    札 B-NS
    幌 E-NS
    雪 O
    国 O
    庙 O
    会 O
    。 O

    主 O
    道 O
    上 O
    的 O
    雪 O

    ...

对于POS和dependency parsing，要求按照CTB9的格式与标签集。

Example:

    1       印度    _       NR      NR      _       3       nn      _       _
    2       海军    _       NN      NN      _       3       nn      _       _
    3       参谋长  _       NN      NN      _       5       nsubjpass       _       _
    4       被      _       SB      SB      _       5       pass    _       _
    5       解职    _       VV      VV      _       0       root    _       _

    1       新华社  _       NR      NR      _       7       dep     _       _
    2       新德里  _       NR      NR      _       7       dep     _       _
    3       １２月  _       NT      NT      _       7       dep     _       _
    ...

该函数有如下参数：
- **data_path**:str，用于微调的数据集文件的路径。
- **task**：str，此次微调的任务，可选值'CWS','POS','Parsing','NER'。
- **lr**：float，微调的学习率。默认取1e-5。
- **n_epochs**：int，微调的迭代次数，默认取1。
- **batch_size**:int，每个batch的数据数量，默认为8。
- **save**:bool，是否保存微调后的模型，默认为False。
- **save_url**:str，若保存模型，则此值为保存模型的路径。

**词典分词**

用户可以使用模型的 add_user_dict 函数添加自定义词典，该词典会影响模型在分词任务中的权重分配。进行分词任务时，首先利用词典进行正向、反向最大匹配法进行分词，并将词典方法的分词结果乘上权重系数融入到深度学习模型的结果中。该函数的参数可以是由词组成的list，也可以是文件路径（文件中的内容是由'\n'分隔开的词）。

用户可使用 set_user_dict_weight 函数设置权重系数（若不设置，默认为0.05）。我们在大规模的训练语料库中发现0.05-0.1即可取得较好的结果。条件允许的情况下，用户也可以自行设置验证集、测试集，找到最适合自己任务的权重系数。

添加完用户词典后，需要在调用模型时令 use_dict 参数为True。

用户可调用 remove_user_dict 移除之前添加的用户词典。

使用用户词典影响分词的一则例子如下：
```
sentence="奥利奥利奥"
print(model(sentence))
model.add_user_dict(["奥利","奥利奥"])
print(model(sentence,use_dict=True))
```
输出为：
```
[['奥利奥利奥']]
[['奥利', '奥利奥']]
```

**分词风格**

分词风格，指的是训练模型中文分词模块的10个语料库，模型可以区分这10个语料库，设置分词style为S即令模型认为现在正在处理S语料库的分词。所以分词style实际上是与语料库的覆盖面、分词粒度相关的。如本模型默认的CTB语料库分词粒度较细。如果想切换不同的粒度，可以使用模型的 set_cws_style 函数，例子如下：

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
对语料库的选取参考了下方CWS SOTA模型的论文，共包括：SIGHAN 2005的 MSR、PKU、AS、CITYU 语料库，由山西大学发布的 SXU 语料库，由斯坦福的CoreNLP 发布的 CTB6 语料库，由国家语委公布的 CNC 语料库，由王威廉先生公开的微博树库 WTB，由张梅山先生公开的诛仙语料库 ZX，Universal Dependencies 项目的 UD 语料库。

**输入与输出**

输入模型的可以是单独的字符串，也可是由字符串组成的列表。如果输入的是列表，模型将一次性处理所有输入的字符串，所以请自行控制 batch size。

模型的输出可以是python的list，也可以是fastHan中自定义的Sentence与Token类。模型默认返回list。

如果将"return_list"参数设为False，模型将输出一个由sentence组成的列表，而每个sentence又由token组成。每个token本身代表一个被分好的词，有pos、head、head_label、ner、loc属性，代表了该词的词性、依存关系、命名实体识别信息、起始位置。

一则输入输出的例子如下所示：

```
sentence=["我爱踢足球。","林丹是冠军"]
answer=model(sentence,'Parsing',return_list=False)
for i,sentence in enumerate(answer):
    print(i)
    for token in sentence:
        print(token,token.pos,token.head,token.head_label)
```
上述代码将输出如下内容：

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

此外，由于各项任务共享词表、词嵌入，即使不切换模型的分词风格，模型对繁体字、英文字母、数字均具有一定识别能力。

## **模型表现**

**泛化能力测试**

对于NLP工具包来说，最重要的就是泛化能力，即在未知数据集里的表现。我们选取了样本较为复杂的Weibo数据集。我们在Weibo的dev集和test集进行了分词测试，并与jieba、THULAC、LTP4.0、SnowNLP进行了对比，对比结果如下（spanF metric）。

 数据集 | SnowNLP | jieba | THULAC | LTP4.0 base | fastHan large | fastHan large (fine-tuned)
--- | --- | --- | --- | --- | --- | ---
Weibo devset|0.7999|0.8319 |0.8649|0.9182|0.9314 |0.9632
Weibo testset|0.7965 | 0.8358 | 0.8665 | 0. 9205 | 0.9338 | 0.9664

作为可以现成使用的工具，fastHan的准确率相较于SnowNLP、jieba、THULAC有较大提升。相较于LTP 4.0-base，fastHan的准确率更高，且模型更小（262MB：492MB）。
在finetune之后，fastHan的准确率也提升明显。

**准确率测试**

模型在以下数据集进行训练和准确性测试：

- CWS：AS, CITYU, CNC, CTB, MSR, PKU, SXU, UDC, WTB, ZX
- NER：MSRA、OntoNotes
- POS & Parsing：CTB9

注：模型在训练NER OntoNotes时将其标签集转换为与MSRA一致。

模型在ctb分词语料库的前800句进行了速度测试，平均每句有45.2个字符。测试环境为私人电脑， Intel Core i5-9400f + NVIDIA GeForce GTX 1660ti，batch size取8。经测试依存分析运行速度较慢，其他各项任务运行速度大致相同。base模型上依存分析任务使用GPU效果不佳，是因为依存分析利用POS的结果需要大量CPU计算，GPU带来的加速效果小于信息传递的负担。

最终模型取得的表现如下：


任务 | CWS | Parsing | POS | NER MSRA | NER OntoNotes | 速度(句/s),cpu|速度(句/s)，gpu
---|---|--- |--- |--- |--- |---|---
SOTA模型 | 97.1 | 85.66,81.71 | 93.15 | 96.09 | 81.82 |——|——
base模型 | 97.27 | 81.22,76.71 | 94.88 | 94.33 | 82.86 |25-55|22-111
large模型 | 97.41 | 85.52,81.38 | 95.66 | 95.50 | 83.82|14-28|21-97

注：模型在句首加入语料库标签来区分输入句子的任务及语料库，最初测试计算F值时将语料库标签也算入在内，导致CWS、POS分值偏高。现在已修复此处错误并更新了表格中的CWS及POS项。评分略有下降（CWS平均下降0.11，POS平均下降0.29），但仍超越SOTA模型。

表格中单位为百分数。CWS的成绩是10项任务的平均成绩。Parsing中的两个成绩分别代表Fudep和Fldep。SOTA模型的数据来自笔者对网上资料及论文的查阅，如有缺漏请指正，不胜感激。这五项SOTA表现分别来自如下五篇论文：

1. Huang W, Cheng X, Chen K, et al. Toward Fast and Accurate Neural Chinese Word Segmentation with Multi-Criteria Learning.[J]. arXiv: Computation and Language, 2019.
2. Hang Yan, Xipeng Qiu, and Xuanjing Huang. "A Graph-based Model for Joint Chinese Word Segmentation and Dependency Parsing." Transactions of the Association for Computational Linguistics 8 (2020): 78-92.
3. Meng Y, Wu W, Wang F, et al. Glyce: Glyph-vectors for Chinese Character Representations[J]. arXiv: Computation and Language, 2019.
4. Xiaonan  Li,  Hang  Yan,  Xipeng  Qiu,  and  XuanjingHuang. 2020. FLAT: Chinese NER using flat-latticetransformer.InProceedings of the 58th AnnualMeeting of the Association for Computational Lin-guistics, pages 6836–6842, Online. Association forComputational Linguisti

