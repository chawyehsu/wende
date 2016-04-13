# -*- coding: utf-8 -*-
"""
    ltml.py
    ~~~~~~~

    The LTML class, LTP-Cloud XML data format. Based on:

    https://github.com/HIT-SCIR/ltp-cloud-api-tutorial/blob/master/Python%2FCustoms%2FLTML.py
"""

from __future__ import unicode_literals
from xml.dom import minidom
from xml.etree import ElementTree
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
from xml.etree.ElementTree import tostring


class LTML(object):
    def __init__(self, xml_str=None):
        if xml_str is not None:
            self.dom = ElementTree.fromstring(xml_str)
        self._xml4nlp = None
        self._note = None
        self._doc = None

    def _set_sent_on_note(self):
        self._note.set("sent", "y")

    def _set_word_on_note(self):
        self._set_sent_on_note()
        self._note.set("word", "y")

    def _set_pos_on_note(self):
        self._set_word_on_note()
        self._note.set("pos", "y")

    def _set_ne_on_note(self):
        self._set_pos_on_note()
        self._note.set("ne", "y")

    def _set_parser_on_note(self):
        self._set_pos_on_note()
        self._note.set("parser", "y")

    def _set_semparser_on_note(self):
        self._set_parser_on_note()
        self._note.set("semparser", "y")

    def _set_srl_on_note(self):
        self._set_semparser_on_note()
        self._note.set("srl", "y")

    def _set_all_on_note(self):
        self._set_srl_on_note()
        self._note.set("ne", "y")

    def _clean_note(self):
        self._note.set("sent", "n")
        self._note.set("word", "n")
        self._note.set("pos", "n")
        self._note.set("ne", "n")
        self._note.set("parser", "n")
        self._note.set("semparser", "n")
        self._note.set("srl", "n")

    def build(self, text):
        """ Build LTML from string
        :param text: the text string
        """
        self._xml4nlp = Element('xml4nlp')
        self._note = SubElement(self._xml4nlp, 'note')
        self._doc = SubElement(self._xml4nlp, 'doc')

        para = SubElement(self._doc, 'para')
        para.set("id", "0")
        para.text = text

        self._clean_note()
        self.dom = self._xml4nlp

    def build_from_words(self, words):
        """ Build LTML from words, automatically detect input type
        :param words: the words, list(str), list(list), list(tuple) is supported
        """
        if isinstance(words, unicode):
            self.build(words)
        elif isinstance(words, list):
            flag = "seg"
            assert len(words) > 0

            word = words[0]
            if isinstance(word, unicode):
                flag = "seg"
            elif ((isinstance(word, list) or isinstance(word, tuple)) and
                  len(word) == 2 and isinstance(word[0], unicode) and isinstance(word[1], unicode)):
                flag = "pos"
            elif ((isinstance(word, list) or isinstance(word, tuple)) and
                  len(word) == 4 and isinstance(word[0], unicode) and isinstance(word[1], unicode)):
                flag = "dp"
            else:
                flag = "unknown"

            self._xml4nlp = Element('xml4nlp')
            self._note = SubElement(self._xml4nlp, 'note')
            self._doc = SubElement(self._xml4nlp, 'doc')

            para = SubElement(self._doc, 'para')
            sent = SubElement(para, 'sent')

            para.set("id", "0")
            sent.set("id", "0")

            self._clean_note()

            if flag == "seg":
                for i, word in enumerate(words):
                    sent.append(Element('word', {
                        'id': unicode(i),
                        'cont': word
                    }))
                sent.set('cont', ("".join(words)))
                self._set_word_on_note()
            elif flag == "pos":
                for i, word_pos in enumerate(words):
                    word, pos = word_pos
                    sent.append(Element('word', {
                        'id': unicode(i),
                        'cont': word,
                        'pos': pos
                    }))
                sent.set('cont', ("".join([word[0] for word in words])))
                self._set_pos_on_note()
            elif flag == "dp":
                for i, rep in enumerate(words):
                    word, pos, head, dep_rel = rep
                    sent.append(Element('word', {
                        'id': unicode(i),
                        'cont': word,
                        'pos': pos,
                        'parent': str(int(head) - 1),
                        'relation': dep_rel
                    }))
                sent.set('cont', ("".join([word[0] for word in words])))
                self._set_parser_on_note()

            self.dom = self._xml4nlp

    def tostring(self, encoding="utf-8"):
        """ Make XML to UTF-8 String.
        Note: LTP-Cloud API needs text to be UTF-8 encoding, or will return `ENCODING NOT IN UTF8` error.
        """
        return tostring(self.dom, encoding)

    def prettify(self):
        """ Return a pretty-printed XML string for the Element.
        """
        re_parsed = minidom.parseString(tostring(self.dom))
        return re_parsed.toprettyxml()
