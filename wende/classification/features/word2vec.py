# -*- coding:utf-8 -*-
from __future__ import unicode_literals
import gensim
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin
from wende.classification.nlp import tokenize
from wende.config import WORD2VEC_MODEL_DIR

w2v_model = gensim.models.Word2Vec.load(WORD2VEC_MODEL_DIR)


def question2vec(words, num_features):

    # remove unseen terms
    words = filter(lambda x: x in w2v_model, words)

    q_vec = np.zeros(num_features, dtype="float32")
    word_count = 0

    for word in words:
        word_count += 1
        q_vec += w2v_model[word]

    word_count = 1 if word_count == 0 else word_count
    q_vec /= word_count
    # print(q_vec)
    return q_vec


def gen_review_vecs(reviews, num_features=200):

    curr_index = 0
    review_feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:

       if curr_index % 1000 == 0.:
           print "Vectorizing review %d of %d" % (curr_index, len(reviews))

       review_feature_vecs[curr_index] = question2vec(review, num_features)
       curr_index += 1

    print(len(review_feature_vecs))

    return review_feature_vecs


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
        """Transform a sequence of documents to a document-word2vec matrix.

        Parameters
        ----------
        X : iterable over raw text documents, length = n_samples
            Samples. Each sample must be a text document (either bytes or
            unicode strings, file name or file object depending on the
            constructor argument) which will be tokenized and hashed.

        y : (ignored)

        Returns
        -------
        X : scipy.sparse matrix, shape = (n_samples, self.n_features)
            Document-term matrix.

        """
        analyzer = self.build_analyzer()
        raw_X = [analyzer(doc) for doc in raw_documents]
        # _X = (Question2VecVectorizer._doc2vec(doc_words) for doc_words in raw_X)
        X = gen_review_vecs(raw_X)

        return X

    # Alias transform to fit_transform for convenience
    fit_transform = transform

