# fastHan
## Brief Introduction
fastHan is developed based on [fastNLP](https://github.com/fastnlp/fastNLP) and pytorch. It is as convinient to use as spacy.

The kernel model of FastHan is based on BERT. We trained the kernel model on 13 corpus of CWS, POS, NER and dependency parsing. There are two versions of fastHan: base(4 layers, 150MB) and large(8 layers, 262mb). The large model can achieve SOTA performance on each task. 


## Install
To install fastHan, the environment has to satisfy requirements below：

- torch>=1.0.0
- fastNLP>=0.5.5

You can execute the following command to complete the installation：

```
pip install fastHan
```

## User Tutorial

It is quite simple to use FastHan. There are two steps: load the model, call the model.

#### Load the model
Execute the following code to load the model:

```
from fastHan import FastHan
model=FastHan()
```

If this is the first time to load the model, fastHan will download parameters from our server automatically.

FastHan will load base model by default. If you want to load large model,  you can set the arg **model_type** 


```
model=FastHan(model_type="large")
```
Besides, for users download parameters manually, load the model by path is also allowed. e.g.:
```
model=FastHan(model_type='large',url="C:/Users/gzc/.fastNLP/fasthan/fasthan_large")
```

#### Call the model
An example of calling the model is shown below:

```
sentence="郭靖是金庸笔下的男主角。"
answer=model(sentence)
print(answer)
answer=model(sentence,target="Parsing")
print(answer)
answer=model(sentence,target="NER")
print(answer)
```
and the output will be：

```
[['郭靖', '是', '金庸', '笔', '下', '的', '男', '主角', '。']]
[[['郭靖', 2, 'top', 'NR'], ['是', 0, 'root', 'VC'], ['金庸', 4, 'nn', 'NR'], ['笔', 5, 'lobj', 'NN'], ['下', 8, 'assmod', 'LC'], ['的', 5, 'assm', 'DEG'], ['男', 8, 'amod', 'JJ'], ['主角', 2, 'attr', 'NN'], ['。', 2, 'punct', 'PU']]]
[[['郭靖', 'NR'], ['金庸', 'NR']]]
```
arg list：
- **target**: the value can be set in ['Parsing'、'CWS'、'POS'、'NER'],and the default value is 'CWS'
  - fastHan uses CTB label set for POS\Parsing, uses MSRA label set for NER
- **use_dict**: whether to use user lexicon，default by False.
- **return_list**：whether to return as list, default by True.
- **return_loc**: whether to return the start position of words, deault by False. It can be used in spanF metric。


**segmentation style**

Segmentation style refers to 10 CWS corpus used in our training phase. Our model can distinguish the corpus and follow the criterion of the corpora. So the segmentation style is related to the critetion of each corpora. Users can use set_cws_style to change the style. e.g.:


>
```
sentence="一个苹果。"
print(model(sentence,'CWS'))
model.set_cws_style('cnc')
print(model(sentence,'CWS'))
```
the output will be:

```
[['一', '个', '苹果', '。']]
[['一个', '苹果', '。']]
```
our corpus include SIGHAN2005(MSR,PKU,AS,CITYU),SXU,CTB6,CNC,WTB,ZX,UD.

**Input and Output**

Input of the model can be string or list consist of strings. If the input is list, model will process the list as one batch. So users need to control the batch size.

Output of the model can be list, or Sentence and Token defined in fastHan. Model will output list by default.

if "return_list" is False, fastHan will output a list consist of Sentence, and Sentence is consist of Token. Each token represents a word after segmentation, and have attributes: pos, head, head_label, ner and loc.

An example is shown as follows:

```
sentence=["我爱踢足球。","林丹是冠军"]
answer=model(sentence,'Parsing',return_list=False)
for i,sentence in enumerate(answer):
    print(i)
    for token in sentence:
        print(token,token.pos,token.head,token.head_label)
```
the output will be:

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

**change device**

Users can use **set_device** function to change the device of model:

```
model.set_device('cuda:0')
model.set_device('cpu')
```

**User lexicon**

Users can use **add_user_dict** to add their own lexicon, which will affect the weight put into CRF. The arg of this function can be list consist of words, or the path of lexicon file. (In the file,words are split by '\n').

Users can use **set_user_dict_weight** to set the weight coefficient of user lexicon, which is default by 0.05. Users can optimize the value by construct dev set.

Users can use **remove_user_dict** to remove the lexicon added before.
```
sentence="奥利奥利奥"
print(model(sentence))
model.add_user_dict(["奥利","奥利奥"])
print(model(sentence,use_dict=True))
```
The output will be:
```
[['奥利奥利奥']]
[['奥利', '奥利奥']]
```
## Performance

### Generalization test
Generalization is the most important attribute for a NLP toolkit. We conducted a CWS test on the dev set of the Weibo dataset, and compared fastHan with jieba, THULAC, LTP4.0, SnowNLP. The results are as follows (spanF metric):


 dataset | SnowNLP | jieba | THULAC | LTP4.0 base | fastHan large
--- | --- | --- | --- | --- | ---
Weibo |0.7999|0.8319 |0.8649|0.9182|0.9314

fastHan's performance is much better than SnowNLP, jieba and THULAC. Compared with LTP-4.0, fastHan's model is much smaller(262MB:492MB) and the scores is 1.3 percentage points higher.


### Accuracy test
We use following corpus to train fastHan and implement accucacy test：

- CWS：AS, CITYU, CNC, CTB, MSR, PKU, SXU, UDC, WTB, ZX
- NER：MSRA、OntoNotes
- POS & Parsing：CTB9

We also perform speed test with Intel Core i5-9400f + NVIDIA GeForce GTX 1660ti, and batch size is set to 8.

Results are as follows:


test | CWS | Parsing | POS | NER MSRA | NER OntoNotes | 速度(句/s),cpu|速度(句/s)，gpu
---|---|--- |--- |--- |--- |---|---
SOTA | 97.1 | 85.66,81.71 | 93.15 | 96.09 | 81.82 |——|——
fastHan base | 97.27 | 81.22,76.71 | 94.88 | 94.33 | 82.86 |25-55|22-111
fastHan large | 97.41 | 85.52,81.38 | 95.66 | 95.50 | 83.82|14-28|21-97

The SOTA results come from following papers:

1. Huang W, Cheng X, Chen K, et al. Toward Fast and Accurate Neural Chinese Word Segmentation with Multi-Criteria Learning.[J]. arXiv: Computation and Language, 2019.
2. Hang Yan, Xipeng Qiu, and Xuanjing Huang. "A Graph-based Model for Joint Chinese Word Segmentation and Dependency Parsing." Transactions of the Association for Computational Linguistics 8 (2020): 78-92.
3. Meng Y, Wu W, Wang F, et al. Glyce: Glyph-vectors for Chinese Character Representations[J]. arXiv: Computation and Language, 2019.
4. Xiaonan  Li,  Hang  Yan,  Xipeng  Qiu,  and  XuanjingHuang. 2020. FLAT: Chinese NER using flat-latticetransformer.InProceedings of the 58th AnnualMeeting of the Association for Computational Lin-guistics, pages 6836–6842, Online. Association forComputational Linguisti

