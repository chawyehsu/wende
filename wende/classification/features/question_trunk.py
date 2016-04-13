# -*- coding:utf-8 -*-
"""
    question_trunk.py
    ~~~~~~~~~~~~~~~~~

    “问题主干”特征
"""

from __future__ import unicode_literals
import fileinput
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from wende.classification.nlp import tokenize, pos_tagging, dependency_parsing
from wende.config import INTERROGATIVE_WORD_FILE

inter_words = [line.strip().decode('utf-8')
               for line in fileinput.input(INTERROGATIVE_WORD_FILE)]
fileinput.close()


class QuestionTrunkVectorizer(TfidfVectorizer):
    """ “问题主干”特征：
            直接选取问题的疑问词、疑问词的一级句法依存词、
            以及问题的主干（主谓宾）作为问题特征
    """
    def __init__(self, tokenizer=tokenize, ngram_range=(1, 1),
                 max_df=1.0, sublinear_tf=False):
        super(QuestionTrunkVectorizer, self).__init__(
            tokenizer=tokenizer, ngram_range=ngram_range,
            max_df=max_df, sublinear_tf=sublinear_tf)

    @staticmethod
    def interrogative_word_extract(words):
        """ 疑问词抽取
        :param words: 题分词结果
        :return: 问题的疑问词列表（取决于人工构建的疑问词表）
        """
        rv = set()
        for word in words:
            if word in inter_words:
                rv.add(word)
        logging.debug("FEATURE_SELECTION:extracted interrogative words: {}"
                      .format(" ".join(list(rv))))
        return rv

    @classmethod
    def _trunk_extract(cls, words):
        """ 特征选择方法
        :param words: 问题分词结果
        :return: “问题主干”特征词列表
        """
        # 获得问题的依存句法树
        dp_tree = dependency_parsing(words, pos_tagging(words))
        # 获取问题的疑问词
        iws = QuestionTrunkVectorizer.interrogative_word_extract(words)
        # 疑问词加入特征集合
        fet_tokens = iws
        for node in dp_tree:
            if node['cont'] in iws:
                # 疑问词的一级依存词
                fet_tokens.add(dp_tree[node['parent']]['cont'])
            # 问题主干：核心词（HED）、核心词的一级依存词
            if (node['relate'] == 'HED') or \
                    (dp_tree[node['parent']]['relate'] == 'HED'):
                fet_tokens.add(node['cont'])
        rv = list(fet_tokens)
        logging.debug("FEATURE_SELECTION:QuestionTrunk features: {}"
                      .format(" ".join(rv)))
        return rv

    def build_analyzer(self):
        # 重载分词器
        tokenize = self.build_tokenizer()
        return lambda doc: self._word_ngrams(
            QuestionTrunkVectorizer._trunk_extract(tokenize(doc)))
