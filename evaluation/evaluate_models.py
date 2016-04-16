# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.learning_curve import learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from wende.classification.model import load_data
from wende.classification.nlp import tokenize
from wende.config import DATASET


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("训练数据量")
    plt.ylabel("准确率")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs,
        scoring='accuracy', train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="训练集")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="测试集（交叉校验）")

    plt.legend(loc="best")
    return plt

# 数据集
dataset = load_data(DATASET)
# print(dataset)
X, y = dataset.data, dataset.target

# 交叉校验 (CV)
# This cross-validation object is a merge of StratifiedKFold and ShuffleSplit,
# which returns stratified randomized folds. The folds are made by preserving
# the percentage of samples for each class.
cv = cross_validation.StratifiedShuffleSplit(y, test_size=0.2, random_state=42)

# 基线特征 (tfidf: baseline feature)
f_tfidf = TfidfVectorizer(tokenizer=tokenize, sublinear_tf=True, max_df=0.5)

# 分类模型
clf_1 = MultinomialNB(alpha=5)
clf_2 = LinearSVC(C=0.02)

# 绘图
title = "学习曲线（朴素贝叶斯）"
model = Pipeline([('feat', f_tfidf), ('clf', clf_1)])
plot_learning_curve(model, title, X, y, ylim=(0.7, 1.01), cv=cv)

title = "学习曲线（支持向量机）"
model = Pipeline([('feat', f_tfidf), ('clf', clf_2)])
plot_learning_curve(model, title, X, y, ylim=(0.7, 1.01), cv=cv)

plt.show()
