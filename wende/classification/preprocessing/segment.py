# -*- coding:utf-8 -*-
""" 预处理：分词模块 """
from __future__ import unicode_literals
import jieba as jbseg
import logging as log
from bosonnlp import BosonNLP
from wende.classification.preprocessing import stopwords
from wende.config import BOSON_API_TOKEN


def cut(text, tokenizer='jieba'):
    """
    :param text: 待分词句子
    :param tokenizer: 分词方式（结巴分词或者波森分词）
    :return: 分词结果（已去除停用词）
    """
    if tokenizer == 'jieba':
        words = ' '.join(jieba(text))
        log.debug("question cut: %s" % words)
        return words
    elif tokenizer == 'boson':
        words = ' '.join(boson(text))
        log.debug("question cut: %s" % words)
        return words
    else:
        raise Exception("Invalid tokenizer type (only 'jieba' or 'boson')")


def jieba(text):
    jbseg.setLogLevel('INFO')
    rv_words = jbseg.lcut(text)
    return remove_stopwords(rv_words)


def boson(text):
    # boson_nlp = BosonNLP(os.getenv('BOSONNLP_KEY'))
    boson_nlp = BosonNLP(BOSON_API_TOKEN)
    rv_words = boson_nlp.tag(text)[0]['word']
    return remove_stopwords(rv_words)


def remove_stopwords(words_list):
    for w in words_list:
        if w in stopwords:
            words_list.remove(w)
    return words_list


if __name__ == "__main__":
    _jb = cut("北京大学的副校长是谁？")
    _bs = cut("北京大学的副校长是谁？", "boson")

    print("{0}: {1}".format("Jieba", ' '.join(_jb)))
    print("{0}: {1}".format("Boson", ' '.join(_bs)))
