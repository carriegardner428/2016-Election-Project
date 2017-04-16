import os
import numpy as np
import re
from time import time
import pandas as pd
from pandas import DataFrame
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import FeatureUnion

from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.svm import SVC


def remove_stops(obs):
    stops = set(stopwords.words('english'))
    tokens = obs.apply(lambda x: [item for item in x if item not in stops])
    print(str(type(tokens)))
    return tokens

def preprocess(df, remove_stopwords=True):
    obs = df.text.str.lower()
    obs = obs.str.replace(r'https?:\/\/.*[\r\n]*', '') # remove http(s) links
    obs = obs.str.replace(r'@[A-Za-z0-9]+', '') # remove mentions
    obs = obs.str.strip()
    obs = obs.str.strip('""')
    obs = obs.apply(nltk.word_tokenize)
    if (remove_stopwords):
        obs = remove_stops(obs)
    obs = obs.str.join(' ')
    tweets['text'] = obs
    return obs


class EstimatorSearch:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=1, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X, y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            z = dict(list(params.items()) + list(d.items()))
            return pd.Series(z)

        rows = [row(k, gsc.cv_validation_scores, gsc.parameters)
                     for k in self.keys
                     for gsc in self.grid_searches[k].grid_scores_]
        df = pd.concat(rows, axis=1).T.sort([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]


############# Main
tweets = pd.read_csv('tweets_by_user.csv')
preprocess(tweets, remove_stopwords=False)
tweets.head(5)

users = pd.read_csv('users.csv')
df = pd.merge(tweets, users, on='userID')

df.head(5)
df = df.dropna()
df.shape

df = df[df.party != 'I']
df.head(5)


X = df['text']
y = df['party']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

tfidf = TfidfVectorizer(stop_words='english', max_df=1)
svd = TruncatedSVD(n_components=100)
normalizer = Normalizer(copy=False)

pipeline = make_pipeline(tfidf, svd, normalizer)

X_train_dtm = pipeline.fit_transform(X)


models = {
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVC': SVC()
}

params = {
    'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
    'RandomForestClassifier': { 'n_estimators': [16, 32] },
    'AdaBoostClassifier':  { 'n_estimators': [16, 32] },
    'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
    'SVC': [
        {'kernel': ['linear'], 'C': [1, 10]},
        {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
    ]
}

search = EstimatorSearch(models, params)
search.fit(X_train_dtm, y, scoring=None, n_jobs=-1)

print(search.score_summary())
