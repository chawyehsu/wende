# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from time import time
from sklearn import cross_validation
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from wende.classification.features import QuestionTrunkVectorizer, Question2VecVectorizer
from wende.classification.model import load_data
from wende.classification.nlp import tokenize
from wende.config import DATASET

# data
dataset = load_data(DATASET)
print(dataset)
X, y = dataset.data, dataset.target

# features and models
f_tfidf = TfidfVectorizer(tokenizer=tokenize, max_df=0.5)
f_tfidf_chi2 = Pipeline([
    ('tfidf', f_tfidf),
    # 降维_特征选择: 卡方检验 (χ2)
    ('chi2_select', SelectKBest(chi2, k=200)),
])
f_tfidf_lsa = Pipeline([
    ('tfidf', f_tfidf),
    # 降维_特征抽取: 潜在语义分析 (LSA)
    ('lsa', TruncatedSVD(n_components=200, n_iter=10))
])

f_trunk = QuestionTrunkVectorizer(tokenizer=tokenize)
f_trunk_chi2 = Pipeline([
    ('trunk', f_trunk),
    # 降维_特征选择: 卡方检验 (χ2)
    ('chi2_select', SelectKBest(chi2, k=200)),
])
f_trunk_lsa = Pipeline([
    ('trunk', f_trunk),
    # 降维_特征抽取: 潜在语义分析 (LSA)
    ('lsa', TruncatedSVD(n_components=200, n_iter=10))
])

# Word2Vec 向量特征
f_word2vec = Question2VecVectorizer(tokenizer=tokenize)

union_f_1 = FeatureUnion([
    ('feats_1', f_trunk),
    ('feats_2', f_word2vec),
])
union_f_2 = FeatureUnion([
    ('f_trunk_lsa', Pipeline([
        ('trunk', f_trunk),
        # 降维_特征抽取: 潜在语义分析 (LSA)
        ('lsa', TruncatedSVD(n_components=200, n_iter=10))
    ])),
    ('feats_2', f_word2vec),
])


clf_1 = MultinomialNB(alpha=.1)
clf_2 = LinearSVC()

model = Pipeline([('feat', f_tfidf), ('clf', clf_1)])

cv = cross_validation.StratifiedKFold(y, n_folds=5)

t0 = time()
f1_score = cross_validation.cross_val_score(model, X, y, cv=cv, scoring='recall_weighted').mean()
print("time cost: {}".format(time() - t0))

# Model                        F1 score        time cost
# --------------------------------------------------------
# Naive Bayes:
# ========================================================
# Feature                      F1 score        time cost
# --------------------------------------------------------
# f_tfidf:                  0.895163554507 / 12.4296369553  |
# f_tfidf_chi2:             0.886842588668 / 12.3169968128  |-- baseline
# f_tfidf_lsa:              0.891201975115 / 15.1656749249  |
# f_trunk:                  0.907255735005 / 74.9544141293
# f_trunk_chi2:             0.902699898010 / 76.9226009846
# f_trunk_lsa:              0.905672947647 / 77.7997789383
# f_word2vec:               0.793071144334 / 12.8787701130
# f_trunk + f_word2vec:     0.932029249449 / 81.7183940411  | may because n_features increase
# --------------------------------------------------------
# f_trunk_lsa + f_word2vec: 0.928265493993 / 90.3578629494  | decrease n_features to 400 by using lsa on trunk feature


print("f1 score: {}".format(f1_score))



