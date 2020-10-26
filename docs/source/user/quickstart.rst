===============
快速入门
===============

这篇教程可以带你从零开始了解 fastHan 的使用。

fastHan的使用流程分为如下两步：

1.初始化模型
~~~~~~~~~~~~

.. code-block:: python

    from fastHan import FastHan
    model=FastHan()

此时若用户为首次初始化模型，将自动从服务器中下载参数。

模型默认初始化为base，如果使用large版本，可在初始化时加入如下参数：

.. code-block:: python

    model=FastHan(model_type="large")

2.将句子输入模型
~~~~~~~~~~~~

.. code-block:: python

    sentence="郭靖是金庸笔下的一名男主。"
    answer=model(sentence,target="Parsing")
    print(answer)
    answer=model(sentence,target="NER")
    print(answer)

模型将会输出如下信息：

.. code-block:: text

    [[['郭靖', 2, 'top', 'NR'], ['是', 0, 'root', 'VC'], ['金庸', 4, 'nn', 'NR'], ['笔', 5, 'lobj', 'NN'], ['下', 10, 'assmod', 'LC'], ['的', 5, 'assm', 'DEG'], ['一', 8, 'nummod', 'CD'], ['名', 10, 'clf', 'M'], ['男', 10, 'amod', 'JJ'], ['主', 2, 'attr', 'NN'], ['。', 2, 'punct', 'PU']]]
    [[['郭靖', 'NR'], ['金庸', 'NR']]]

此外，模型拥有如下这些功能：

任务选择
~~~~~~~~~~~~

target参数可在'Parsing'、'CWS'、'POS'、'NER'四个选项中取值，模型将分别进行依存分析、分词、词性标注、命名实体识别任务,模型默认进行CWS任务。其中词性标注任务包含了分词的信息，而依存分析任务又包含了词性标注任务的信息。命名实体识别任务相较其他任务独立。

如果分别运行CWS、POS、Parsing任务，模型输出的分词结果等可能存在冲突。如果想获得不冲突的各类信息，请直接运行包含全部所需信息的那项任务。

模型的POS、Parsing任务均使用CTB标签集。NER使用msra标签集。


分词风格
~~~~~~~~~~~~
分词风格，指的是训练模型中文分词模块的10个语料库，模型可以区分这10个语料库，设置分词style为S即令模型认为现在正在处理S语料库的分词。所以分词style实际上是与语料库的覆盖面、分词粒度相关的。如本模型默认的CTB语料库分词粒度较细。如果想切换不同的粒度，可以使用模型的set_cws_style函数，例子如下：

.. code-block:: python

    sentence="一个苹果。"
    print(model(sentence,'CWS'))
    model.set_cws_style('cnc')
    print(model(sentence,'CWS'))

模型将输出如下内容：

.. code-block:: text

    [['一', '个', '苹果', '。']]
    [['一个', '苹果', '。']]

对语料库的选取参考了下方CWS SOTA模型的论文，共包括：SIGHAN 2005的 MSR、PKU、AS、CITYU 语料库，由山西大学发布的 SXU 语料库，由斯坦福的CoreNLP 发布的 CTB6 语料库，由国家语委公布的 CNC 语料库，由王威廉先生公开的微博树库 WTB，由张梅山先生公开的诛仙语料库 ZX，Universal Dependencies 项目的 UD 语料库。

输入与输出
~~~~~~~~~~~~
输入模型的可以是单独的字符串，也可是由字符串组成的列表。如果输入的是列表，模型将一次性处理所有输入的字符串，所以请自行控制 batch size。

模型的输出是在fastHan模块中定义的sentence与token类。模型将输出一个由sentence组成的列表，而每个sentence又由token组成。每个token本身代表一个被分好的词，有pos、head、head_label、ner四项属性，代表了该词的词性、依存关系、命名实体识别信息。

一则输入输出的例子如下所示：

.. code-block:: python

    sentence=["我爱踢足球。","林丹是冠军"]
    answer=model(sentence,'Parsing')
    for i,sentence in enumerate(answer):
        print(i)
        for token in sentence:
            print(token,token.pos,token.head,token.head_label)

模型将输出如下内容：

.. code-block:: text

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

可在分词风格中选择'as'、'cityu'进行繁体字分词，这两项为繁体语料库。

此外，由于各项任务共享词表、词嵌入，即使不切换模型的分词风格，模型对繁体字、英文字母、数字均具有一定识别能力。

切换设备
~~~~~~~~~~~~
可使用模型的 set_device 函数，令模型在cuda上运行或切换回cpu，示例如下：

.. code-block:: python

    model.set_device('cuda:0')
    model.set_device('cpu')


词典分词
~~~~~~~~~~~~
用户可以使用模型的 add_user_dict 函数添加自定义词典，该词典会影响模型在分词任务中的权重分配。进行分词任务时，首先利用词典进行正向、反向最大匹配法进行分词，并将词典方法的分词结果乘上权重系数融入到深度学习模型的结果中。该函数的参数可以是由词组成的list，也可以是文件路径（文件中的内容是由'\n'分隔开的词）。

用户可使用 set_user_dict_weight 函数设置权重系数（若不设置，默认为0.05）。我们在大规模的训练语料库中发现0.05-0.1即可取得较好的结果。条件允许的情况下，用户也可以自行设置验证集、测试集，找到最适合自己任务的权重系数。

添加完用户词典后，需要在调用模型时令 use_dict 参数为True。再次申明，词典功能目前仅在'CWS'任务中有效。

用户可调用 remove_user_dict 移除之前添加的用户词典。

使用用户词典影响分词的一则例子如下：

.. code-block:: python

    sentence="奥利奥利奥"
    print(model(sentence))
    model.add_user_dict(["奥利","奥利奥"])
    model.set_user_dict_weight(0.05)
    print(model(sentence,use_dict=True))

输出为：

.. code-block:: text

    [['奥利奥利奥']]
    [['奥利', '奥利奥']]
    