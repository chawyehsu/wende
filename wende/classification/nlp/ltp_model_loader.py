# -*- coding:utf-8 -*-
"""
    ltp_model_loader.py
    ~~~~~~~~~~~~~~~~~~~

    LTP 模型加载器
"""

from __future__ import unicode_literals
import logging
import os
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer
from time import time
from wende.config import LTP_DATADIR


def load(obj):
    # 分词
    if isinstance(obj, Segmentor):
        logging.info("loading ltp segment model...")
        t0 = time()
        obj.load((os.path.join(LTP_DATADIR, "cws.model")).encode('utf-8'))
        logging.info("ltp segment model loading done in {}s."
                     .format(time() - t0))

    # 词性标注
    if isinstance(obj, Postagger):
        logging.info("loading ltp pos tagging model...")
        t0 = time()
        obj.load((os.path.join(LTP_DATADIR, "pos.model")).encode('utf-8'))
        logging.info("ltp pos tagging model loading done in {}s."
                     .format(time() - t0))

    # 句法分析
    if isinstance(obj, Parser):
        logging.info("loading ltp parser model...")
        t0 = time()
        obj.load((os.path.join(LTP_DATADIR, "parser.model")).encode('utf-8'))
        logging.info("ltp parser model loading done in {}s."
                     .format(time() - t0))

    # 命名实体标注
    if isinstance(obj, NamedEntityRecognizer):
        logging.info("loading ltp ner model...")
        t0 = time()
        obj.load((os.path.join(LTP_DATADIR, "ner.model")).encode('utf-8'))
        logging.info("ltp ner model loading done in {}s."
                     .format(time() - t0))
