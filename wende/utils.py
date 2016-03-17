# -*- coding:utf-8 -*-
from __future__ import unicode_literals
from os import path

USERASK = path.join(path.abspath(path.dirname(__name__)), 'classification/data/user_ask.txt')


def save_userask(qtype, question, qcut):
    """ 保存用户的问题，可为后面做训练集增量训练
    :param qtype: 判断到问题类型（不一定为正确类型）
    :param question: 用户输入的原问题
    :param qcut: 分词后的问题
    """
    with open(USERASK, 'a') as collector:
        collector.write(b"{0},{1},{2}\n".format(qtype.encode('utf-8'), question.encode('utf-8'), qcut.encode('utf-8')))


if __name__ == "__main__":
    save_userask('HUM', '中山大学 的 副校长 是 谁', '中山大学的副校长是谁？')
