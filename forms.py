from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField, EmailField, SelectField, RadioField
from wtforms.validators import DataRequired, URL, Email
from flask_ckeditor import CKEditorField
import requests


class InfoForm (FlaskForm):
    address = StringField(u"Input any location")
    node = StringField(u"Input any node nearby specified location")
    model_field = SelectField(u"Select Machine Learning Model to Plot:",
                              choices=["linear",
                                       "lasso",
                                       "ridge",
                                       "bayesian_ridge",
                                       "polynomial",
                                       "decision_tree",
                                       "random_forest",
                                       "gbr_quantile",
                                       "gbr_mse",
                                       "xgboost"])
    submit = SubmitField("Submit")