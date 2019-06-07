'''
IML HACKATHON 19' - Tweets Challenge
View README for explanation
'''

import pandas as pd
import glob
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import model_selection
import os
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re

# text cleaning
TWEETS_COL = 'tweet'

BAD_CHARS = re.compile('[;,"/\(\){}\[\]\|]')
DIVIDE_AT = re.compile('@(.+)')
DIVIDE_HASHTAG = re.compile('#(.+)')
DIVIDE_HTTP = re.compile('http([^ ]*)')
EMOJIS = re.compile(r'([\u263a-\U0001f645])')
PUNCTUATION = re.compile('[,.]')
ALL_CAPS = re.compile(r'(\b[A-Z][A-Z0-9]+\b)')



def main():
    data, tags = load_data()

    # clean tweet
    data[TWEETS_COL] = data[TWEETS_COL].apply(clean_text)

    # Split data
    trainX, testX, trainY, testY = model_selection.train_test_split (data['tweet'], data['user'],
                                                                     test_size=0.2, shuffle=True)
    # model trainer and model selection
    model, testX, testY = learner(trainX, testX, trainY, testY, tags)

    y_pred = model.predict(testX)
    accuracy = accuracy_score(y_pred, testY)
    print('accuracy from file: %s' % accuracy)


def clean_text(data):
    '''
    :param data: String in the dataset (tweets)
    '''
    result = re.search(ALL_CAPS, data)
    if result:
        data = ALL_CAPS.sub(" allcaps123 " + result.group(1), data)
    data = data.lower() # lowercase text
    result = re.search (DIVIDE_AT, data)
    data = EMOJIS.sub(' emoji123 ', data, count=0)
    if "@" in data:
        data = DIVIDE_AT.sub('@ '+result.group(1), data, count=0)
    result = re.search (DIVIDE_HASHTAG, data)
    if result:
        data = DIVIDE_HASHTAG.sub('# '+result.group(1), data, count=0)
    data = BAD_CHARS.sub('', data, count=0)
    data = re.compile(' +').sub(' ', data, count=0)
    data = EMOJIS.sub(' emoji123 ', data, count=0)
    data = PUNCTUATION.sub(' punc123 ', data, count=0)
    data = re.compile(' +').sub(' ', data, count=0)
    data = re.compile ('\'').sub ('', data, count=0)
    data = data + ' ' + str(len(data))
    return data


def learner(trainX, testX, trainY, testY, tags):
    '''
    Can apply Naive bayes, SVM and logsitic regression.
    After testing, SVM reaches the best result.
    '''
    naive_bayes (trainX, trainY, testX, testY, tags)
    logistic_regression (trainX, trainY, testX, testY, tags)
    acc, model = support_vector_machine (trainX, trainY, testX, testY, tags)
    return model, testX, testY


def load_data():
    # Load all csv's to one file
    path = 'tweets_data'
    all_files = glob.glob (path + os.path.sep + "*.csv")
    # names of classes
    tags = pd.read_csv (path + os.path.sep + 'names.txt', index_col=None, header=None)
    tags = tags[1].tolist ()
    all_data =[]
    for filename in all_files:
        if filename != ("tweets_data"+ os.path.sep+"tweets_test_demo.csv"):
            df = pd.read_csv (filename, index_col=None, header=0)
            all_data.append(df)
    data = pd.concat (all_data, axis=0, ignore_index=True)
    return data, tags


# NAIVE BAYES - Multinomial
def naive_bayes(X_train, y_train, X_test, y_test, tags):
    nb = Pipeline([('vect', CountVectorizer ()),
                    ('tfidf', TfidfTransformer ()),
                    ('clf', MultinomialNB ()),
                    ])
    return fit_and_predict(X_train, y_train, X_test, y_test, tags, nb)


# SVM with SGD
def support_vector_machine(X_train, y_train, X_test, y_test, tags):
    # without CV
    lin_vec_machine = Pipeline([
                                ('vect', CountVectorizer(ngram_range=(1,2), analyzer='word')),
                                ('tfidf', TfidfTransformer()),
                                ('clf',SGDClassifier(loss='hinge', penalty='elasticnet',
                                                     alpha=1e-5, random_state=41,n_jobs=-1,
                                                     max_iter=500, tol=None)),
                               ])

    sample_report = fit_and_predict(X_train, y_train, X_test, y_test, tags, lin_vec_machine)[1]
    return fit_and_predict(X_train, y_train, X_test, y_test, tags, lin_vec_machine), lin_vec_machine


# Logistic regression with solver=saga and Vectorization
def logistic_regression(X_train, y_train, X_test, y_test, tags):
    log_reg = Pipeline([
                    ('vect', CountVectorizer(ngram_range=(1,2), analyzer='word')),
                    ('tfidf', TfidfTransformer()),
                    # ('feat_select', SelectKBest(k = 30000)),
                    ('clf', LogisticRegression(n_jobs=1, C=1e6, multi_class='multinomial', solver='saga')),
                    ])
    return fit_and_predict(X_train, y_train, X_test, y_test, tags, log_reg)


def fit_and_predict(X_train, y_train, X_test, y_test, tags, classifier):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    print('accuracy %s' % accuracy)
    class_report = classification_report(y_test, y_pred, target_names=tags)
    print(class_report)
    return accuracy, class_report


if __name__ == "__main__":
    main()
