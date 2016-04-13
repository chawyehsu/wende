# -*- coding: utf-8 -*-
""" 分类模型 """
from __future__ import print_function, unicode_literals
import argparse
import fileinput
import logging
from os import path
import re
from time import time
import numpy
from sklearn import cross_validation
from sklearn.datasets.base import Bunch
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
import sys

from .features import QuestionTrunkVectorizer, Question2VecVectorizer
from .nlp import tokenize
from wende.config import DATASET, APP_MODEL_DIR


class Classifier(object):
    """ 分类器
    """
    def __init__(self, init_data=None, model_file="classifier.pkl"):
        self.data = init_data
        self.model_file = model_file
        self.model = self.init_model()

    @staticmethod
    def init_model(clf='svm'):
        """ 分类模型初始化
        """
        # 多项式朴素贝叶斯
        if clf == 'nb':
            clf = MultinomialNB(alpha=10)
        # 线性支持向量机
        else:
            clf = LinearSVC(C=0.01)

        # “问题主干”特征
        f_trunk = QuestionTrunkVectorizer(tokenizer=tokenize)

        # Word2Vec 向量特征
        f_word2vec = Question2VecVectorizer(tokenizer=tokenize)

        # 最终联合特征 (400维)
        union_features = FeatureUnion([
            ('f_trunk_lsa', Pipeline([
                ('trunk', f_trunk),
                # 降维_特征抽取: 潜在语义分析 (LSA)
                ('lsa', TruncatedSVD(n_components=200, n_iter=10))
            ])),
            ('f_word2vec', f_word2vec),
        ])

        model = Pipeline([('union', union_features), ('clf', clf)])
        return model

    def train_model(self):
        logging.debug("training model...")
        t0 = time()
        self.model.fit(self.data.data, self.data.target)
        print("training time cost: {}".format(time() - t0))

    def save_model(self, model_file=None):
        """ Save model to file with joblib's replacement of pickle. See:
        http://scikit-learn.org/stable/tutorial/basic/tutorial.html#model-persistence
        """
        if not model_file:
            model_file = self.model_file
        logging.debug("saving model to file: " + model_file)
        joblib.dump(self.model, path.join(APP_MODEL_DIR, model_file))

    def load_model(self, model_file=None):
        """ 从文件载入分类模型
        :param model_file: 模型文件名称
        :return: self
        """
        if not model_file:
            model_file = self.model_file
        self.model = joblib.load(path.join(APP_MODEL_DIR, model_file))
        logging.debug("loaded model from file: " + model_file)
        return self

    def predict(self, text):
        """ 预测问题类型
        :param text: 要进行预测的问题
        :return: 问题类型
        """
        qtype = self.model.predict([text])[0]
        logging.debug("predict: {}".format(qtype))
        return qtype

    def test_model(self, n_folds=10):
        """ 使用 `分层K-折交叉校验（Stratified K-folds cross-validating）`
            对模型进行测试
        """
        logging.debug("testing model with {}-folds CV".format(n_folds))
        model = self.init_model()
        X = self.data.data
        y = self.data.target

        cv = cross_validation.StratifiedKFold(y, n_folds=n_folds)

        # 准确率
        precision = cross_validation.cross_val_score(
            model, X, y, cv=cv, scoring='precision_weighted').mean()
        # 召回率
        recall = cross_validation.cross_val_score(
            model, X, y, cv=cv, scoring='recall_weighted').mean()
        # F1
        t0 = time()
        f1 = cross_validation.cross_val_score(
            model, X, y, cv=cv, scoring='f1_weighted').mean()
        t = time() - t0

        print("-" * 92)
        print("precision: {}".format(precision))
        print("recall: {}".format(recall))
        print("f1: {}".format(f1))
        print("testing time cost: {}".format(t))


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
                            .format(fileinput.filename(),
                                    fileinput.filelineno()))

        label, text = match.group(1), match.group(2)

        if label not in target_names:
            target_names[label] = len(target_names)
        # 使用原始的分类标签（`HUM`, `LOC`, etc.）
        target.append(label)
        # 使用数字索引做为分类标签（{'HUM': 1, 'LOC': 2}）
        # target.append(target_names[label])
        data.append(text)

    return Bunch(
        data=numpy.array(data),
        target=numpy.array(target),
        target_names=numpy.array([k for k in target_names]),
    )


if __name__ == "__main__":
    # Usage: `python -m wende.classification.model [args]`
    parser = argparse.ArgumentParser(
        description='Question type classification')
    parser.add_argument("-t", "--test",
                        help="test the classifier",
                        action="store_true")
    parser.add_argument("-s", "--save",
                        help="save the trained model to file",
                        action="store_true")
    parser.add_argument("-f", "--savefile",
                        help="the filename where the model should be saved")
    parser.add_argument("-p", "--predict",
                        help="classify an input question")
    args = parser.parse_args()

    # Load dataset
    data = load_data(DATASET)
    print(data)

    if args.test:
        clf = Classifier(data)
        clf.test_model(n_folds=5)
        sys.exit(0)

    if args.save:
        clf = Classifier(data)
        clf.train_model()
        if args.savefile:
            clf.save_model(model_file=args.savefile)
        else:
            clf.save_model()
        sys.exit(0)

    if args.predict:
        clf = Classifier()
        clf.load_model()
        print(clf.predict(args.predict))
        sys.exit(0)

    print("Nothing to do...")
    sys.exit(1)
