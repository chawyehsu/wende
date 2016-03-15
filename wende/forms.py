# -*- coding:utf-8 -*-
from flask.ext.wtf import Form
from wtforms import StringField
from wtforms import SubmitField
from wtforms.validators import DataRequired


class QuestionForm(Form):
    question = StringField('Question', validators=[DataRequired()])
    submit_button = SubmitField('Go')
