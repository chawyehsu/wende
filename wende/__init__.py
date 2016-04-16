# -*- coding: utf-8 -*-
""" 问答系统 """
from __future__ import unicode_literals
import json
from flask import Flask, request, render_template
from flask.ext.bootstrap import Bootstrap
import logging as log
from wende.classification import model
from wende.classification.nlp import tokenize, keywords_extract
from wende.config import SECRET_KEY, LOGGING
from wende.forms import QuestionForm
from wende.utils import save_userask


def answer_question(question):
    """问题 -> 分词 -> 分类 -> 答案抽取 -> ...
        |---> 关键词提取 -->|
    """
    log.info("answering question: '{}'".format(question))
    # tokens_with_tags = tokenize_with_pos_tag(question)
    qcut = tokenize(question)
    # 对分词后的问句进行分类
    qtype = classify_question(question)
    # 问题关键词提取
    # kwords = extract_keywords(question)
    # return qtype, qcut, "; ".join(kwords)
    return qtype, qcut, None


def classify_question(question):
    """ 载入分类模型并对给定的问题进行分类
    :param question: 输入的问题（原始问题，不需做分词预处理）
    :return: 问题类型
    """
    log.debug("classifying question...")
    clf = model.Classifier().load_model()
    qtype = clf.predict(question)
    log.info("classified question type: '{}'".format(qtype))
    return qtype


def extract_keywords(question):
    """
    :param question: 输入的问题
    :return: 问题关键词列表（带权重）
    """
    log.debug("extracting keywords...")
    rv = keywords_extract(question)

    kwords = []
    for word, weight in rv:
        kwords.append("{0}: {1}".format(word, weight))
    log.debug("; ".join(kwords))

    return kwords


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = SECRET_KEY
    Bootstrap(app)

    @app.route('/', methods=('GET', 'POST'))
    def index():
        if request.method == 'POST':
            question = request.form['question']
            qtype, qcut, kwords = answer_question(question)
            # 保存用户的问题及判断到的类型
            save_userask(qtype, question, qcut)
            return render_template(
                'answer.html', qtype=qtype,
                qcut=" ".join(qcut), kwords=kwords,
                question=question)

        return render_template('index.html', form=QuestionForm())

    # API
    @app.route('/api/classify', methods=('GET', 'POST'))
    def api_classify():
        if request.method == 'POST':
            question = request.values.get('q', None)
            qtype, qcut, kwords = answer_question(question)
            save_userask(qtype, question, qcut)
            return json.dumps({'error': '0', 'question': question, 'qtype': qtype})

        return json.dumps({'error': '405', 'message': 'wrong request method'})

    return app
