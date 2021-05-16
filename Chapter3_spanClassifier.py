import tarfile
import os
import pandas as pd
import numpy as np

UNCOMPRESS_DIR = './datasets/spam/Uncompressed/'
uncompress = False

files = ['./datasets/spam/20021010_easy_ham.tar.bz2',
        './datasets/spam/20021010_hard_ham.tar.bz2',
        './datasets/spam/20021010_spam.tar.bz2',
        './datasets/spam/20030228_easy_ham.tar.bz2',
        './datasets/spam/20030228_easy_ham_2.tar.bz2',
        './datasets/spam/20030228_hard_ham.tar.bz2',
        './datasets/spam/20030228_spam.tar.bz2',
        './datasets/spam/20030228_spam_2.tar.bz2',
        './datasets/spam/20050311_spam_2.tar.bz2']

if(uncompress):
    for f in files:
        print('Uncompressing: ', f)
        tar = tarfile.open(f, 'r:bz2')
        tar.extractall('./datasets/spam/Uncompressed/')
        tar.close()

df = pd.DataFrame(columns = ['RawData', 'isSpam'])

for root, dirs, files in os.walk(UNCOMPRESS_DIR):
    for d in dirs:
        for root2, dirs2, files2 in os.walk(UNCOMPRESS_DIR + '/' + d):
            for f in files2:
                isSpam = 0
                if(d.__contains__('spam')):
                    isSpam = 1
                try:
                    data = open(UNCOMPRESS_DIR + '/' + d + '/' + f, 'r').read()
                    df = df.append({'RawData': data, 'isSpam':isSpam}, ignore_index=True)
                    df.head()
                except:
                    print(f)
                    continue

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

X = df.RawData
y = df.isSpam

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(X,y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

from sklearn.base import BaseEstimator, TransformerMixin
import string
from email import parser
import email
import regex as re
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 toLowerCaseFlag=True,
                 removeHeadersFlag=True,
                 removePunctuationFlag=True,
                 replaceURLsFlag=True,
                 replaceNumbersFlag=True,
                 doStemmingFlag=False,
                 removeNewLinesFlag=True):
        self.toLowerCaseFlag = toLowerCaseFlag
        self.removeHeadersFlag = removeHeadersFlag
        self.removePunctuationFlag = removePunctuationFlag
        self.replaceURLsFlag = replaceURLsFlag
        self.doStemmingFlag = doStemmingFlag
        self.replaceNumbersFlag = replaceNumbersFlag
        self.removeNewLinesFlag = removeNewLinesFlag

    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        XX = np.array(X, copy=True)
        for i in range(0,len(XX)):
            if(self.removeHeadersFlag):
                try:
                    XX[i] = self.removeHeaders(XX[i])
                except:
                    print(i)
                    print(X[i])
                    print(self.removeHeaders(XX[i]))
            if(self.replaceURLsFlag):
                XX[i] = self.replaceURLs(XX[i])
            if(self.replaceNumbersFlag):
                XX[i] = self.replaceNumbers(XX[i])
            if(self.removePunctuationFlag):
                XX[i] = self.removePunctuation(XX[i])
            if(self.toLowerCaseFlag):
                XX[i] = self.toLowerCase(XX[i])
            if(self.doStemmingFlag):
                XX[i] = self.stemEmail(XX[i])
            if(self.removeNewLinesFlag):
                XX[i] = self.removeNewLines(XX[i])

        return XX

    def toLowerCase(self, email):
        return email.lower()

    def removePunctuation(self, s):
        return s.translate(str.maketrans('', '', string.punctuation))

    def removeHeaders(self, s):
        msg = email.message_from_string(s)
        return_str = ''
        for part in msg.walk():
            # each part is a either non-multipart, or another multipart message
            # that contains further parts... Message is organized like a tree
            if part.get_content_type() == 'text/plain':
                return_str +=  part.get_payload()  + ' '# prints the raw text
        return return_str

    def replaceURLs(self, s):
        return re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' URL ', s, flags=re.MULTILINE)

    def replaceNumbers(self, s):
        return re.sub('\d+', ' NUM ' , s, flags=re.MULTILINE)

    def removeNewLines(self, s):
        return re.sub('\n', ' ' , s, flags=re.MULTILINE)

    def stemEmail(self, s):
        ps = PorterStemmer()
        words = word_tokenize(s)
        return_str = ''
        for w in words:
            return_str += ps.stem(w) + ' '
        return return_str

from sklearn.pipeline import Pipeline

preprocessing_pipeline = Pipeline([
    ('text_processing', DataTransformer()),
    ('text_vectorizer', TfidfVectorizer())
                                  ])

from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


knn_pipeline = Pipeline([
    ('text_processing', DataTransformer()),
    ('text_vectorizer', TfidfVectorizer()),
    ('estimator', KNeighborsClassifier())])

rf_pipeline  = Pipeline([
    ('text_processing', DataTransformer()),
    ('text_vectorizer', TfidfVectorizer()),
    ('estimator', RandomForestClassifier())])

svc_pipeline = Pipeline([
    ('text_processing', DataTransformer()),
    ('text_vectorizer', TfidfVectorizer()),
    ('estimator', SVC())])

svc_rbf_pipeline  = Pipeline([
    ('text_processing', DataTransformer()),
    ('text_vectorizer', TfidfVectorizer()),
    ('estimator', SVC(kernel='rbf'))])

gnb_pipeline  = Pipeline([
    ('text_processing', DataTransformer()),
    ('text_vectorizer', TfidfVectorizer()),
    ('estimator', GaussianNB())])

import nltk
from nltk.corpus import stopwords

set(stopwords.words('english'))

svc_params = {
    #'text_processing__replaceNumbersFlag':[True, False],
    'text_processing__doStemmingFlag':[True, False],
    #'text_processing__removeHeadersFlag':[True, False],
    'text_processing__toLowerCaseFlag':[True, False],
    'text_vectorizer__lowercase':[False],
    #'text_processing__replaceNumbersFlag':[True,False],
    'text_vectorizer__max_df':[0.25,1],
    'text_vectorizer__min_df':[0.01,1],
    'text_vectorizer__max_features':[1000,10000, None],
    'text_vectorizer__stop_words':[stopwords.words('english'), None],
    'estimator__C' : [0.1,1.0,10.0],
    'estimator__kernel':['poly', 'rbf', 'sigmoid']
    }

print(svc_params)


# test - If this works --- It Works!!!
print('testing pipeline')
rf_pipeline.fit(X_train, y_train.values.astype(np.uint8))
print('score: ', rf_pipeline.score(X_test,  y_test.values.astype(np.uint8)))
print('pipeline fitted')

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(svc_pipeline,
                           param_grid=svc_params,
                           cv=2,
                           scoring='neg_mean_squared_error',
                           return_train_score=True,
                           n_jobs=-1,
                           verbose=2)


print('starting grid search')

grid_search.fit(X_train, y_train.values.astype(np.uint8))

print('Success')

print(grid_search.cv_results_)
print(grid_search.best_params_)
