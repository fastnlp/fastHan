===============
安装指南
===============

.. contents::
   :local:

fastHan 依赖如下包::

    torch>=1.0.0
    fastNLP>=0.5.0

.. note::

    其中torch的安装可能与操作系统及 CUDA 的版本相关，请参见 `PyTorch 官网 <https://pytorch.org/>`_ 。
    此外，如果使用0.5.0版本的fastNLP，建议使用1.0.0版本的torch，否则在解码阶段会有bug影响准确率。如果使用高版本的torch，请使用0.5.5版本的fastNLP。

..  code:: shell

   >>> pip install fastHan
