# -*- coding: utf-8 -*-
""" 系统配置 """
import logging
import logging.config

DEBUG = True

# BosonNLP API TOKEN, 见：http://docs.bosonnlp.com/
BOSON_API_TOKEN = ''

SECRET_KEY = 'devkey12345'


LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'default': {
            'format': '%(asctime)s %(levelname)s %(module)s: %(message)s'
        },
        'simple': {
            'format': '[%(levelname)8s (%(module)s)] -- %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
        # 'file': {
        #     'level': 'DEBUG',
        #     'class': 'logging.RotatingFileHandler',
        #     'formatter': 'default',
        #     'filename': 'wende.log',
        #     'maxBytes': 1024,
        #     'backupCount': 3
        # },
    },
    'loggers': {
        '': {
            # 'handlers': ['file', 'console'],
            'handlers': ['console'],
            'level': 'DEBUG',
        },
    }
}
logging.config.dictConfig(LOGGING)
