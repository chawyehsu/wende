# -*- coding:utf-8 -*-
"""
    pos_tagging.py
    ~~~~~~~~~~~~~~

    NLP: 词性标注


    # 词性标注集及含义 (From HIT-LTP:
                        https://ltp.readthedocs.org/zh_CN/latest/appendix.html)

    LTP使用的是863词性标注集，其各个词性含义如下表。

    +-----+---------------------+------------+-----+-------------------+------------+
    | Tag | Description         | Example    | Tag | Description       | Example    |
    +=====+=====================+============+=====+===================+============+
    | a   | adjective           | 美丽       | ni  | organization name | 保险公司   |
    +-----+---------------------+------------+-----+-------------------+------------+
    | b   | other noun-modifier | 大型, 西式 | nl  | location noun     | 城郊       |
    +-----+---------------------+------------+-----+-------------------+------------+
    | c   | conjunction         | 和, 虽然   | ns  | geographical name | 北京       |
    +-----+---------------------+------------+-----+-------------------+------------+
    | d   | adverb              | 很         | nt  | temporal noun     | 近日, 明代 |
    +-----+---------------------+------------+-----+-------------------+------------+
    | e   | exclamation         | 哎         | nz  | other proper noun | 诺贝尔奖   |
    +-----+---------------------+------------+-----+-------------------+------------+
    | g   | morpheme            | 茨, 甥     | o   | onomatopoeia      | 哗啦       |
    +-----+---------------------+------------+-----+-------------------+------------+
    | h   | prefix              | 阿, 伪     | p   | preposition       | 在, 把     |
    +-----+---------------------+------------+-----+-------------------+------------+
    | i   | idiom               | 百花齐放   | q   | quantity          | 个         |
    +-----+---------------------+------------+-----+-------------------+------------+
    | j   | abbreviation        | 公检法     | r   | pronoun           | 我们       |
    +-----+---------------------+------------+-----+-------------------+------------+
    | k   | suffix              | 界, 率     | u   | auxiliary         | 的, 地     |
    +-----+---------------------+------------+-----+-------------------+------------+
    | m   | number              | 一, 第一   | v   | verb              | 跑, 学习   |
    +-----+---------------------+------------+-----+-------------------+------------+
    | n   | general noun        | 苹果       | wp  | punctuation       | ，。！     |
    +-----+---------------------+------------+-----+-------------------+------------+
    | nd  | direction noun      | 右侧       | ws  | foreign words     | CPU        |
    +-----+---------------------+------------+-----+-------------------+------------+
    | nh  | person name         | 杜甫, 汤姆 | x   | non-lexeme        | 萄, 翱     |
    +-----+---------------------+------------+-----+-------------------+------------+

"""

from __future__ import unicode_literals
import logging
from pyltp import Postagger
from . import ltp_model_loader

__all__ = ['pos_tagging']

# 加载模型
postagger = Postagger()
ltp_model_loader.load(postagger)


def pos_tagging(words):
    """ 词性标注
    :param words: 分词结果
    :return: 词性标注列表
    """
    postags = postagger.postag([i for i in words])
    rv = [i for i in postags]
    logging.debug("NLP:pos tagging: {}".format(" ".join(rv)))
    return rv
