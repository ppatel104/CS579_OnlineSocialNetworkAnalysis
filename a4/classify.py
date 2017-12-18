"""
classify.py
"""

import pandas as pd
import nltk as nk
import numpy as np
from sklearn.feature_extraction import stop_words
from collections import Counter,defaultdict
from nltk.stem import PorterStemmer,LancasterStemmer,SnowballStemmer
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import re
import enchant
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import warnings

def read_data():
    tweets = pd.read_csv("united_tweets.csv")
    return tweets

def data_preprocessing(text):
    sws = stopwords.words("english")
    with open("stanford_stopwords") as f:
        stanford_sw = f.readlines()
    stanford_sw = [x.strip() for x in stanford_sw]
    sw = list(set(sws + stanford_sw + list(stop_words.ENGLISH_STOP_WORDS)))
    d = enchant.Dict("en_US")
    sb = SnowballStemmer("english")
    l = []
    for i in range(0, len(text['tweet'])):
        tweet = [x.lower() for x in nk.word_tokenize(re.sub(r"http\S+", "", text['tweet'].iloc[i])) if x.isalpha() and d.check(x) and x not in sw]
        # tweet = x for x in tweet if x not in sw)
        stem_words = [sb.stem(x) for x in tweet]
        seen = set()
        result = []
        for item in tweet:
            if sb.stem(item) not in seen:
                seen.add(sb.stem(item))
                result.append(item)
        tweet = ' '.join(x for x in result)
        l.append(tweet)
    return l

def featurize():
    warnings.filterwarnings("ignore")
    tweets = read_data()
    new_tweets = data_preprocessing(tweets)
    tweets['tweet'] = new_tweets
    tfidf = TfidfVectorizer(ngram_range=(1,3),min_df=5)
    data = tfidf.fit_transform(tweets['tweet']).toarray()
    sb = SnowballStemmer("english")
    stem_features = [sb.stem(x) for x in tfidf.get_feature_names()]
    a = list(tfidf.get_feature_names())
    seen = set()
    result = []
    for item in a:
        if sb.stem(item) not in seen:
            seen.add(sb.stem(item))
            result.append(item)
    df1 = pd.DataFrame(data,columns = tfidf.get_feature_names())
    df2 = df1[result]
    df2['sentiment'] = tweets['sentiment'].values
    return df2

def model_fitting():
    data = featurize()
    X_train,X_test,Y_train,Y_test = train_test_split(data.drop('sentiment',axis=1),data['sentiment'],random_state=367)
    results = []
    SVM = svm.LinearSVC(class_weight='balanced', random_state=367, penalty='l1', dual=False)
    SVM.fit(X_train,Y_train)
    prediction = SVM.predict(X_test)
    accuracy = accuracy_score(Y_test,prediction)
    f1 = f1_score(Y_test,prediction)
    return accuracy,f1

def main():

    accuracy,f1 = model_fitting()
    data = read_data()
    labels = list(data['sentiment'])
    c = Counter()
    c.update(labels)

    print ("Number of Positive Instances: %d"%c[1])
    print ("Number of Negative Instances: %d"%c[0])
    print ("--------------------------------------")
    print ("Accuracy: %f"%accuracy)
    print ("F1 Score: %f"%f1)
    print ("---------------------------------------")
    print ("Positive Class Example: %s"%str(data['tweet'][data.sentiment==1].iloc[0]))
    print ("Negatvie Class Example: %s"%str(data['tweet'][data.sentiment==0].iloc[3]))
    f = open('classify.txt','w')
    f.write('%d\n'%c[1])
    f.write('%d\n'%c[0])
    f.write('%s\n'%str(data['tweet'][data.sentiment==1].iloc[0]))
    f.write('%s'%str(data['tweet'][data.sentiment==0].iloc[3]))
    f.close()

if __name__ == main():
    print ("----------------------classify.py-------------------------")
    main()
    print ("----------------------classify.py-------------------------")