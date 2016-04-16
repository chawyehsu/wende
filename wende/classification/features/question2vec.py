# -*- coding:utf-8 -*-
from __future__ import unicode_literals
import logging
import gensim
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin
from wende.classification.nlp import tokenize
from wende.config import WORD2VEC_MODEL_DIR, WORD2VEC_MODEL_SIZE

logging.info("loading word2vec model...")
w2v_model = gensim.models.Word2Vec.load(WORD2VEC_MODEL_DIR)


def gen_doc_vec(words, num_features):
    # remove unseen terms
    words = filter(lambda x: x in w2v_model, words)

    doc_vec = np.zeros(num_features, dtype="float32")
    word_count = 0

    for word in words:
        word_count += 1
        doc_vec += w2v_model[word]

    word_count = 1 if word_count == 0 else word_count
    doc_vec /= word_count
    return doc_vec


def gen_docs_vecs(docs, num_features=WORD2VEC_MODEL_SIZE):
    curr_index = 0
    docs_vecs = np.zeros((len(docs), num_features), dtype="float32")

    for doc in docs:
        if curr_index % 1000 == 0.:
            logging.debug("Vectorizing doc {0} of {1}".format(curr_index, len(docs)))

        docs_vecs[curr_index] = gen_doc_vec(doc, num_features)
        curr_index += 1

    return docs_vecs


class Question2VecVectorizer(BaseEstimator, VectorizerMixin):

    def __init__(self, tokenizer=tokenize):
        self.tokenizer = tokenizer

    def build_analyzer(self):
        return lambda doc: self.tokenizer(doc)

    def fit(self, raw_documents, y=None):
        """Pass to transform function"""
        # triggers a parameter validation
        self.transform(raw_documents, y=y)
        return self

    def transform(self, raw_documents, y=None):
        analyzer = self.build_analyzer()
        raw_X = [analyzer(doc) for doc in raw_documents]
        X = gen_docs_vecs(raw_X)

        return X

    # Alias transform to fit_transform for convenience
    fit_transform = transform
