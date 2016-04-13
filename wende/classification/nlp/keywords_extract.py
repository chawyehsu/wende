# -*- coding:utf-8 -*-
"""
    keywords_extract.py
    ~~~~~~~~~~~~~~~~~~~

    NLP: 关键词提取
"""

from __future__ import unicode_literals
import jieba
import jieba.analyse
from .tokenize import stopwords

__all__ = ['keywords_extract']


def keywords_extract(question):
    jieba.analyse.set_stop_words(stopwords)
    rv = jieba.analyse.extract_tags(question, topK=10, withWeight=True)

    return rv

if __name__ == '__main__':
    k_words = keywords_extract("清华大学的副校长是谁？")
    print("jieba:")
    for w, v in k_words:
        print("{0}: {1}".format(w, v))
