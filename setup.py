#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    reqs = f.read()

pkgs = [p for p in find_packages()]
print(pkgs)

setup(
    name='fastHan',
    version='2.0',
    url='https://github.com/fastnlp/fastHan',
    description=(
        '使用深度学习联合模型，解决中文分词、词性标注、命名实体识别、依存分析任务。'
    ),
    long_description=readme,
    long_description_content_type='text/markdown',
    author='王鹏宇',
    license='Apache License',
    python_requires='>=3.6',
    packages=pkgs,
    install_requires=reqs.strip().split('\n'),
)

