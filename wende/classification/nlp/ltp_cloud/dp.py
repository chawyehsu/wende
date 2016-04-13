# -*- coding:utf-8 -*-
""" LTP-CLOUD 依存句法分析
See: http://www.ltp-cloud.com/intro/#dp_how

[附录A] 依存句法关系
-----------------

+------------+-----+----------------------------+----------------------------+
| 关系类型   | Tag | Description                | Example                    |
+============+=====+============================+============================+
| 主谓关系   | SBV | subject-verb               | 我送她一束花 (我 <-- 送)   |
+------------+-----+----------------------------+----------------------------+
| 动宾关系   | VOB | 直接宾语，verb-object      | 我送她一束花 (送 --> 花)   |
+------------+-----+----------------------------+----------------------------+
| 间宾关系   | IOB | 间接宾语，indirect-object  | 我送她一束花 (送 --> 她)   |
+------------+-----+----------------------------+----------------------------+
| 前置宾语   | FOB | 前置宾语，fronting-object  | 他什么书都读 (书 <-- 读)   |
+------------+-----+----------------------------+----------------------------+
| 兼语       | DBL | double                     | 他请我吃饭 (请 --> 我)     |
+------------+-----+----------------------------+----------------------------+
| 定中关系   | ATT | attribute                  | 红苹果 (红 <-- 苹果)       |
+------------+-----+----------------------------+----------------------------+
| 状中结构   | ADV | adverbial                  | 非常美丽 (非常 <-- 美丽)   |
+------------+-----+----------------------------+----------------------------+
| 动补结构   | CMP | complement                 | 做完了作业 (做 --> 完)     |
+------------+-----+----------------------------+----------------------------+
| 并列关系   | COO | coordinate                 | 大山和大海 (大山 --> 大海) |
+------------+-----+----------------------------+----------------------------+
| 介宾关系   | POB | preposition-object         | 在贸易区内 (在 --> 内)     |
+------------+-----+----------------------------+----------------------------+
| 左附加关系 | LAD | left adjunct               | 大山和大海 (和 <-- 大海)   |
+------------+-----+----------------------------+----------------------------+
| 右附加关系 | RAD | right adjunct              | 孩子们 (孩子 --> 们)       |
+------------+-----+----------------------------+----------------------------+
| 独立结构   | IS  | independent structure      | 两个单句在结构上彼此独立   |
+------------+-----+----------------------------+----------------------------+
| 核心关系   | HED | head                       | 指整个句子的核心           |
+------------+-----+----------------------------+----------------------------+
"""

from __future__ import unicode_literals
import json
import logging
from wende.classification.nlp import pos_tagging, tokenize
from wende.classification.nlp.ltp_cloud.base import ltp_request
from wende.classification.nlp.ltp_cloud.ltml import LTML

__all__ = ['dp_online']


def dp_online(words, postags):
    """ 使用在线 ltp-cloud 进行句法分析
    :param words: 分词结果
    :param postags: 词性标注结果
    :return: 句法分析树
    """
    ltml = LTML()
    ltml.build_from_words([(word, tag) for word, tag in zip(words, postags)])
    # logging.debug(ltml.prettify())
    # See: http://www.ltp-cloud.com/document/#api_rest_format_json
    rv = json.loads(ltp_request(ltml.tostring(), 'dp', 'json'))[0][0]
    return rv


def sdp_online(tokens_with_tags):
    """ 对给定问句做语义依存分析
    :param tokens_with_tags: 带有词性标注的分词序列
    :return: json 格式的语义依存树
    """
    ltml = LTML()
    ltml.build_from_words([(pair.word, pair.flag) for pair in tokens_with_tags])
    # rv = ltp_request(ltml.tostring(), 'sdp', 'plain')
    # See: http://www.ltp-cloud.com/document/#api_rest_format_json
    rv = json.loads(ltp_request(ltml.tostring(), 'sdp', 'json'))[0][0]
    logging.debug("LTP sdp return: {}".format(json.dumps(rv)))
    return rv


if __name__ == "__main__":
    words = tokenize("北京大学的副校长是谁？")
    dp_online(words, pos_tagging(words))
