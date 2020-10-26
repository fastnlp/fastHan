===============
版本更新
===============

.. contents::
   :local:

fastHan的版本修正如下问题::

    1.1版本的fastHan与0.5.5版本的fastNLP会导致importerror。如果使用1.1版本的fastHan，请使用0.5.0版本的fastNLP。
    1.2版本的fastHan修复了fastNLP版本兼容问题。小于等于1.2版本的fastHan在输入句子的首尾包含空格、换行符时会产生BUG。如果字符串首尾包含上述字符，请使用strip函数处理输入字符串。
    1.3版本的fastHan自动对输入字符串做strip函数处理。
    1.4版本的fastHan加入用户词典功能（仅限于分词任务）