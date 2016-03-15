# -*- coding:utf-8 -*-
""" 预处理：关键词提取模块 """
from __future__ import unicode_literals
from bosonnlp import BosonNLP
import jieba
import jieba.analyse as jbextractor
from os import path
from wende.config import BOSON_API_TOKEN

stopwords_path = path.join(path.abspath(path.dirname(__file__)), 'data/stopmarks.txt')
jieba.setLogLevel('INFO')


def extract(question, extractor='jieba'):
    """ 提取问题关键词
    :param qcut: 需要进行关键词提取操作的问题
    :param extractor: 调用的关键词提取方法（使用结巴接口 `jieba` 或者波森接口 `boson`）
    :return: 关键词列表，包含权重信息
    """
    if extractor == 'jieba':
        jbextractor.set_stop_words(stopwords_path)
        rv = jbextractor.extract_tags(question, topK=10, withWeight=True)
    elif extractor == 'boson':
        # boson_nlp = BosonNLP(os.getenv('BOSONNLP_KEY'))
        boson_nlp = BosonNLP(BOSON_API_TOKEN)
        rv = boson_nlp.extract_keywords(question, top_k=10)
        # Reverse key and value the BosonAPI returns
        rv = [(i[1], i[0]) for i in rv]
    else:
        raise Exception("Invalid extractor type (only 'jieba' or 'boson')")

    return rv

if __name__ == '__main__':
    _jb = extract("清华大学的副校长是谁？")
    _bs = extract("清华大学的副校长是谁？", 'boson')
    print("jieba:")
    for w, v in _jb:
        print("{0}: {1}".format(w, v))
    print("boson:")
    for w, v in _bs:
        print("{0}: {1}".format(w, v))
