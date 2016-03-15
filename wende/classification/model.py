#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" 分类模型 """
from __future__ import unicode_literals
import argparse
import fileinput
import logging as log
from os import path
import re
import numpy
from sklearn.cross_validation import LeaveOneOut, StratifiedKFold
from sklearn.datasets.base import Bunch
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
import sys

# 模型存取路径
MODEL_DIR = path.join(path.abspath(path.dirname(__file__)), "data/models")


class Classifier(object):
    """ 分类器
    """
    def __init__(self, init_data=None, model_file="classify.pkl"):
        self.data = init_data
        self.model_file = model_file
        self.model = self.init_model()

    @staticmethod
    def init_model():
        """ 初始化分类模型（管道）
        """
        model = Pipeline([
            ('union', FeatureUnion([
                # 词频特征；单字词也统计
                ('words_count', CountVectorizer(token_pattern='(?u)\\b\\w+\\b')),
                # TFIDF 特征；单字词也统计
                ('words_tfidf', TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b'))
            ])),
            # 使用线性 SVM 分类器
            ('clf', LinearSVC()),
        ])
        return model

    def train_model(self):
        """ 训练分类模型
        """
        log.debug("training model...")
        self.model.fit(self.data.data, self.data.target)

    def save_model(self, model_file=None):
        """ Save model to file with joblib's replacement of pickle. See:
        http://scikit-learn.org/stable/tutorial/basic/tutorial.html#model-persistence
        """
        if not model_file:
            model_file = self.model_file
        log.debug("saving model to file: " + model_file)
        joblib.dump(self.model, path.join(MODEL_DIR, model_file))

    def load_model(self, model_file=None):
        """ 从文件载入分类模型
        :param model_file: 模型文件名称
        :return: self
        """
        if not model_file:
            model_file = self.model_file
        log.debug("loading model from file: " + model_file)
        self.model = joblib.load(path.join(MODEL_DIR, model_file))
        return self

    def predict(self, text):
        """ 预测问题类型
        :param text: 要进行预测的问题
        :return: 问题类型
        """
        rv = self.model.predict([text])
        log.debug(rv)
        qtype = self.model.predict([text])[0]
        return qtype

    def test_model(self, n_folds=10, leave_one_out=False):
        """ Test the model by cross-validating with Stratified k-folds
        """
        log.debug("testing model ({} folds)".format(n_folds))
        X = self.data.data
        y = self.data.target
        avg_score = 0.0

        if leave_one_out:
            cv = LeaveOneOut(len(y))
        else:
            cv = StratifiedKFold(y, n_folds=n_folds)

        for train, test in cv:
            model = self.init_model().fit(X[train], y[train])
            avg_score += model.score(X[test], y[test])

        if leave_one_out:
            avg_score /= len(y)
        else:
            avg_score /= n_folds

        print("average score: {}".format(avg_score))
        return avg_score


def load_data(filenames):
    """ 载入数据集
    :param filenames: 数据集文件名
    :return: Bunch 数据对象 See:
    http://scikit-learn.org/stable/datasets/index.html#datasets
    """
    data = []
    target = []
    target_names = {}
    # e.g. HUM,广外 校长 是 谁
    data_re = re.compile(r'(\w+),(.+)')

    for line in fileinput.input(filenames):
        d = data_re.match(line)
        if not d:
            raise Exception("Invalid format in file {} at line {}"
                            .format(fileinput.filename(), fileinput.filelineno()))
        label, text = d.group(1), d.group(2)
        if label not in target_names:
            target_names[label] = len(target_names)
        # 使用数字做为分类标签
        # target.append(target_names[label])
        # 使用原分类标签
        target.append(label)
        data.append(text.decode('utf-8'))

    return Bunch(
        data=numpy.array(data),
        target=numpy.array(target),
        target_names=numpy.array([k for k in target_names]),
    )


if __name__ == "__main__":
    # run from parent dir with `python -m classification.model [args]`
    parser = argparse.ArgumentParser(description='Question type classification')
    parser.add_argument("-t", "--test", help="test the classifier", action="store_true")
    parser.add_argument("-s", "--save", help="save the trained model to disk", action="store_true")
    parser.add_argument("-f", "--savefile", help="the file where the model should be saved")
    parser.add_argument("-p", "--predict", help="classify an input question")
    args = parser.parse_args()

    samples = path.join(path.dirname(__file__), "data/cn_trainset_seg.txt")
    data = load_data(samples)
    # print(data)

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
