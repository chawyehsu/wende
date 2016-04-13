# -*- coding:utf-8 -*-
from __future__ import unicode_literals
from wende.config import USER_ASK


def save_userask(qtype, question, qcut=None):
    """ 保存用户的问题，可为后面做训练集增量训练
    :param qtype: 判断到问题类型（不一定为正确类型）
    :param question: 用户输入的原问题
    :param qcut: 分词后的问题
    """
    with open(USER_ASK, 'ab') as collector:
        collector.write(("{0},{1}\n".format(qtype, question)).encode('utf-8'))


if __name__ == "__main__":
    save_userask('HUM', '中山大学 的 副校长 是 谁', '中山大学的副校长是谁？')
