# -*- coding: utf-8 -*-
"""
========================
Plotting Learning Curves
========================

On the left side the learning curve of a naive Bayes classifier is shown for
the digits dataset. Note that the training score and the cross-validation score
are both not very good at the end. However, the shape of the curve can be found
in more complex datasets very often: the training score is very high at the
beginning and decreases and the cross-validation score is very low at the
beginning and increases. On the right side we see the learning curve of an SVM
with RBF kernel. We can see clearly that the training score is still around
the maximum and the validation score could be increased with more training
samples.
"""
from __future__ import print_function, unicode_literals
import fileinput
from pprint import pprint
import re
from sklearn.datasets.base import Bunch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from wende.classification.features import QuestionTrunkVectorizer, Question2VecVectorizer
from wende.classification.nlp import tokenize

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
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
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def load_data(filenames):
    """ 载入训练模型用的数据集
    :param filenames: 数据集文件名
    :return: Bunch 数据对象. See:
    http://scikit-learn.org/stable/datasets/index.html#datasets
    """
    # 训练问句文本
    data = []
    # 训练问句的实际分类标签
    target = []
    # 分类标签
    target_names = {}
    # 数据集每一行的格式形如：HUM,广外校长是谁?
    data_re = re.compile(r'(\w+),(.+)')

    for line in fileinput.input(filenames):
        match = data_re.match(line.decode('utf-8'))
        if not match:
            raise Exception("Invalid format in dataset {} at line {}"
                            .format(fileinput.filename(), fileinput.filelineno()))

        label, text = match.group(1), match.group(2)
        # 对训练集中的问句进行分词
        # text = " ".join([unicode(i) for i in tokenize_with_pos_tag(text)])

        if label not in target_names:
            target_names[label] = len(target_names)
        # 使用原始的分类标签（`HUM`, `LOC`, etc.）
        target.append(label)
        # 使用数字索引做为分类标签（{'HUM': 1, 'LOC': 2}）
        # target.append(target_names[label])
        data.append(text)

    return Bunch(
        data=np.array(data),
        target=np.array(target),
        target_names=np.array([k for k in target_names]),
    )

digits = load_data('/Users/hanabi/workspace/wende/data/dataset.txt')
print(digits)
X, y = digits.data, digits.target
# print(y)

# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
# cv = cross_validation.ShuffleSplit(digits.data.shape[0], n_iter=10, test_size=0.2, random_state=0)
cv = cross_validation.StratifiedShuffleSplit(y, test_size=0.2, random_state=0)


f_baseline = TfidfVectorizer(tokenizer=tokenize, max_df=0.5)

f_tfidf_lsa = Pipeline([
    ('tfidf', f_baseline),
    # 降维_特征选择: 卡方检验 (χ2, chi-square test)
    # ('chi2_select', SelectKBest(chi2, k=1000)),
    # 降维_特征抽取: 潜在语义分析 (LSI/LSA)
    ('lsi', TruncatedSVD(n_components=200, n_iter=10))
])

f_trunk = QuestionTrunkVectorizer(tokenizer=tokenize)
f_trunk_chi2 = Pipeline([
    ('tfidf', f_trunk),
    # 降维_特征选择: 卡方检验 (χ2, chi-square test)
    ('chi2_select', SelectKBest(chi2, k=200)),
])

# Word2Vec 向量特征
f_word2vec = Question2VecVectorizer(tokenizer=tokenize)

# 联合特征集，亦是最后“喂”给分类器的特征集合
union_f_1 = FeatureUnion([
    ('feats_1', f_trunk),
    ('feats_2', f_word2vec),
])


clf_1 = MultinomialNB(alpha=10)
clf_2 = LinearSVC(C=0.01)

title = "Learning Curves (MultinomialNB(alpha=10))"
model = Pipeline([('feat', f_baseline), ('clf', clf_1)])
plot_learning_curve(model, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

title = "Learning Curves (LinearSVC(C=0.01))"
model = Pipeline([('feat', f_baseline), ('clf', clf_2)])
plot_learning_curve(model, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

plt.show()
