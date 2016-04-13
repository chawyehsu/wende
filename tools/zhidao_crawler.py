#!/usr/bin/env python2
# -*- coding:utf-8 -*-

""" 百度知道问题标题爬取脚本
"""

from __future__ import print_function, unicode_literals
from bs4 import BeautifulSoup
import urllib
import requests
import time

SAVING_TXT = '../data/zhidao.txt'


def zd_request(cid=105, pn=30, keyword=None):
    """
    :param cid: 百度知道问题分栏 id
    :param pn: 分页 [30, 60, 90, 120, 150, 180, ..., ]
    :param keyword: 关键词筛选
    :return: 问题标题列表
    """
    base_uri = "http://zhidao.baidu.com/list?"

    params = {
        'keyWord': keyword.encode('utf-8'),
        'ie': 'utf8',
        'cid': cid,
        'rn': 30,
        'pn': pn,
        # '_pjax': '%23j-question-list-pjax-container',
    }
    params = urllib.urlencode(params)

    headers = {
        "Accept-Encoding": "gzip",
        "Accept-Language": "zh-CN,zh;q=0.8",
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Referer': 'http://zhidao.baidu.com/list?cid=104',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/49.0.2623.87 Safari/537.36',
    }
    # print("headers: {}".format(json.dumps(headers)))
    # print()

    time.sleep(0.6)
    r = requests.get(url=base_uri, params=params, headers=headers)
    print(r.url)
    print("=" * 92)

    resp = r.content.decode('gbk')
    doc = BeautifulSoup(resp, "html.parser")
    links = doc.find_all('a', class_="title-link")
    questions = []
    for l in links:
        questions.append(l.text.strip())

    rv = "\n".join(questions)
    print(rv)
    return rv


if __name__ == "__main__":
    # zd_request(cid=109, keyword="时候")
    with open(SAVING_TXT, 'ab') as zhidao:
        for i in range(30, 510, 30):
            j = zd_request(cid=108, keyword="几点 开播", pn=i).encode('utf-8')
            zhidao.write(b"%s\n" % j)
