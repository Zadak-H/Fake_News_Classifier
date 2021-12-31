# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 20:49:46 2021

@author: Rajarshi
"""

import pandas as pd
df = pd.read_csv('fake-news/train.csv')
df.head()

## Getting the independent features
x = df.drop('label',axis=1)
x.head()

## Getting the dependent features
y=df['label']
y.head()

df.shape

#imporing BOW, TFIDF, HASINGVECTOR
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

#Removing the NAN values from df
df = df.dropna()

msg = df.copy()
msg.reset_index(inplace = True) # dreop ing NaN th indexes need to be rearange
msg['title'][6]

# using corpus to remove the stopwords
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus=[]
for i in range(0, len(msg)):
    rvw = re.sub('[^a-zA-Z]', ' ', msg['title'][i])
    rvw = rvw.lower()
    rvw = rvw.split()
    rvw = [ps.stem(word) for word in rvw if not word in stopwords.words('english')]
    rvw = ' '.join(rvw)
    corpus.append(rvw)

## Applying CountVectorizer and developing BOW model
cv = CountVectorizer(max_features=5000, ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()
X.shape

y = msg['label']

## Train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=0)

cv.get_feature_names()[:20]
cv.get_params()

count_df = pd.DataFrame(X_train, columns = cv.get_feature_names())
count_df.head()

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False,title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearset',cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tricks_marks = np.arrange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        print("Normalized Confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j, i, cm[i,j], horizontallignment = "center", color = "white" if cm[i,j]>thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicated label')
    
# Classifier model
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()

from sklearn import metrics
import numpy as np
import itertools

classifier.fit(X_train,y_train)
pred = classifier.predict(X_test)
score = metrics.confusion_matrix(y_test, pred)
print("accuracy:  %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

# Passive Aggressive Classifier Algorithm

    