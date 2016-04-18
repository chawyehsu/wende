# -*- coding: utf-8 -*-
"""
    app.py
    ~~~~~~

    Wende Chinese QA system
"""

from __future__ import unicode_literals
from wende import create_app
from wende.config import WEB_APP_DEBUG

wende = create_app()

if __name__ == '__main__':
    wende.run(host='127.0.0.1', port=9191, debug=WEB_APP_DEBUG)
