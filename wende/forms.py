# -*- coding:utf-8 -*-
from __future__ import unicode_literals
from flask.ext.wtf import Form
from wtforms import StringField
from wtforms import SubmitField
from wtforms.validators import DataRequired


class QuestionForm(Form):
    question = StringField('Question', validators=[DataRequired()],
                           filters=[lambda x: x.strip() if x is not None else None])
    submit_button = SubmitField('Go')
