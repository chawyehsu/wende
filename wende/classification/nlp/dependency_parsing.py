# -*- coding:utf-8 -*-
"""
    dependency_parsing.py
    ~~~~~~~~~~~~~~~~~~~~~

     NLP: 句法分析

    # 依存句法关系 (From HIT-LTP:
                        https://ltp.readthedocs.org/zh_CN/latest/appendix.html)

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
import logging
from pyltp import Parser
from . import ltp_model_loader
from .ltp_cloud import dp_online

__all__ = ['dependency_parsing']

# 加载模型
parser = Parser()
ltp_model_loader.load(parser)


def dependency_parsing(words, postags, online=False):
    """ 句法分析
    :param words: 分词结果
    :param postags: 词性标注结果
    :return: 句法分析树
    """
    # online=True, 使用 ltp-cloud 做句法分析
    if online:
        return dp_online(words, postags)

    # 使用本地 ltp 做句法分析
    arcs = parser.parse([i.encode('utf-8') for i in words],
                        [i.encode('utf-8') for i in postags])

    # 将句法树封装为类似于 ltp-cloud http 返回 json 的结果
    rv = [{"cont": words[idx],
           "parent": arc.head - 1,
           "relate": arc.relation} for idx, arc in enumerate(arcs)]
    logging.debug("NLP:dependency parsing: {}"
                  .format(" ".join(["{0}:{1}:{2}"
                                   .format(arc['cont'],
                                           arc['parent'],
                                           arc['relate']) for arc in rv])))
    return rv


if __name__ == "__main__":
    dependency_parsing(["北京大学", "的", "副校长", "是", "谁"],
                       ["ni", "u", "n", "v", "r"], online=False)
