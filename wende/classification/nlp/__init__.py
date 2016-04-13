# -*- coding:utf-8 -*-
""" NLP """

from __future__ import absolute_import, unicode_literals
from .dependency_parsing import dependency_parsing
from .keywords_extract import keywords_extract
from .named_entity_recognize import named_entity_recognize
from .pos_tagging import pos_tagging
from .tokenize import tokenize

__all__ = ['tokenize',
           'pos_tagging',
           'dependency_parsing',
           'named_entity_recognize',
           'keywords_extract']
