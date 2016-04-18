# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
import os
from time import time
from sklearn import cross_validation
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from wende.classification.features import QuestionTrunkVectorizer, Question2VecVectorizer
from wende.classification.model import load_data
from wende.classification.nlp import tokenize
from wende.config import DATASET


# 数据集
dataset = load_data(DATASET)
# print(dataset)
X, y = dataset.data, dataset.target


def cross_predict(feat, f_name, X=X, y=y):

    if os.name == 'nt':
        n_jobs = 1
    else:
        n_jobs = -1
    # 分类模型
    # clf_1 = MultinomialNB(alpha=5)
    clf_2 = LinearSVC(C=0.02)

    # 交叉校验 (CV)
    # This cross-validation object is a merge of StratifiedKFold and ShuffleSplit,
    # which returns stratified randomized folds. The folds are made by preserving
    # the percentage of samples for each class.
    #
    #  Note: like the ShuffleSplit strategy, stratified random splits do not guarantee
    # that all folds will be different, although this is still
    # very likely for sizeable datasets.
    #
    # Pass this to cross_val_predict will raise
    # ValueError:cross_val_predict only works for partitions
    #
    # 该 cv 方法不能保证 fold 与 fold 之间的数据不重叠
    # cv = cross_validation.StratifiedShuffleSplit(y, test_size=0.2, random_state=42)

    # This cross-validation object is a variation of KFold that returns stratified folds.
    # The folds are made by preserving the percentage of samples for each class.
    cv = cross_validation.StratifiedKFold(y, n_folds=5, random_state=42)

    model = Pipeline([('feat', feat), ('clf', clf_2)])
    t0 = time()
    y_pred = cross_validation.cross_val_predict(model, X=X, y=y, n_jobs=n_jobs, cv=cv)
    t = time() - t0
    print("=" * 20, f_name, "=" * 20)
    print("time cost: {}".format(t))
    # print("y_predict: {}".format(y_pred))
    print()
    print('confusion matrix:\n', confusion_matrix(y, y_pred))
    print()
    print('\t\taccuracy: {}'.format(accuracy_score(y, y_pred)))
    print()
    print("\t\tclassification report")
    print("-" * 52)
    print(classification_report(y, y_pred))


# 特征
# 基线特征 (tfidf: baseline feature)
f_tfidf = TfidfVectorizer(tokenizer=tokenize, max_df=0.5)
f_tfidf_lsa = Pipeline([
    ('tfidf', f_tfidf),
    # 降维_特征抽取: 潜在语义分析 (LSA)
    ('lsa', TruncatedSVD(n_components=400, n_iter=10))
])

# “问题主干”特征
f_trunk = QuestionTrunkVectorizer(tokenizer=tokenize)
f_trunk_lsa = Pipeline([
    ('trunk', f_trunk),
    # 降维_特征抽取: 潜在语义分析 (LSA)
    ('lsa', TruncatedSVD(n_components=400, n_iter=10))
])

# Word2Vec 平均词向量特征
f_word2vec = Question2VecVectorizer(tokenizer=tokenize)

# 联合特征
union_f_1 = FeatureUnion([
    ('feats_1', f_trunk),
    ('feats_2', f_word2vec),
])
# 联合特征，含降维
union_f_2 = FeatureUnion([
    ('f_trunk_lsa', Pipeline([
        ('trunk', f_trunk),
        # 降维_特征抽取: 潜在语义分析 (LSA)
        ('lsa', TruncatedSVD(n_components=200, n_iter=10))
    ])),
    ('feats_2', f_word2vec),
])

# Do evaluation
# Tfidf 原始维数
cross_predict(f_tfidf, f_name='f_tfidf')
# Tfidf LSA 降维至 400 维
cross_predict(f_tfidf_lsa, f_name='f_tfidf_lsa')
# Trunk 原始维数
cross_predict(f_trunk, f_name='f_trunk')
# Trunk LSA 降维至 400 维
cross_predict(f_trunk_lsa, f_name='f_trunk_lsa')
# Word2Vec 200 维
cross_predict(f_word2vec, f_name='f_word2vec')
# Trunk 原始维数 + Word2Vec 200 维
cross_predict(union_f_1, f_name='union_f_1')
# Trunk LSA 200 维 + Word2Vec 200 维
cross_predict(union_f_2, f_name='union_f_2')
