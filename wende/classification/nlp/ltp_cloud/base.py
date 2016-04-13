# -*- coding:utf-8 -*-
""" LTP-CLOUD HTTP 请求头 """

from __future__ import unicode_literals
import requests
import time
from wende.config import LTP_API_TOKEN


def ltp_request(text, pattern, format='plain'):
    """ LTP-Cloud RESTful 分析接口的封装，此方法不应直接调用
    各参数具体要求见：http://www.ltp-cloud.com/document/#api_rest_param
    :param text: 待分析的文本
    :param pattern: 指定分析模式
    :return:
    """
    _uri_base = "http://api.ltp-cloud.com/analysis/?"
    _data = {
        'api_key': LTP_API_TOKEN,
        'text': text,
        'pattern': pattern,
        'format': format,
        "xml_input": "true",
    }

    # LTP-CLOUD 有请求限制，短时间内多次请求ip会被加入黑洞
    time.sleep(0.8)
    return requests.post(_uri_base, _data).text
