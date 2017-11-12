from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import *


class FeatureSelector(TransformerMixin, BaseEstimator):
    def __init__(self, cols=[]):
        self.cols = cols

    def transform(self, X, **transform_params):
        tmp = X[self.cols]
        return tmp.values

    def fit(self, X, y=None, **fit_params):
        return self


class FeatureNormalizer(TransformerMixin, BaseEstimator):
    def transform(self, X, **transform_params):
        X = (X - X.mean(0)) / X.std(0)
        return X

    def fit(self, X, y=None, **fit_params):
        return self
