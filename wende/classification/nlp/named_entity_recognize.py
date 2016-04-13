# -*- coding:utf-8 -*-
"""
    named_entity_recognize.py
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    NLP: 命名实体识别
"""

from __future__ import unicode_literals
import logging
from pyltp import NamedEntityRecognizer
from . import ltp_model_loader

__all__ = ['named_entity_recognize']

# 加载模型
recognizer = NamedEntityRecognizer()
ltp_model_loader.load(recognizer)


def named_entity_recognize(words, postags):
    """ 命名实体识别
    :param words: 分词结果
    :param postags: 词性标注结果
    :return: 命名实体识别结果
    """
    netags = recognizer.recognize([i.encode('utf-8')for i in words],
                                  [i.encode('utf-8') for i in postags])
    # TODO: 后处理 LTP 返回的实体识别结果，即找出命名实体的具体词语，再返回给调用者
    logging.debug("NLP:named entity recognize: {}"
                  .format(" ".join(netags)))
    pass


if __name__ == "__main__":
    named_entity_recognize(["北京大学", "的", "副校长", "是", "谁"],
                           ["ni", "u", "n", "v", "r"])
