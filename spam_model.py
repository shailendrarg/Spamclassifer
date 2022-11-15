import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

import re #regular expression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer# for lamatisation which creates root words
import pickle

df=pd.read_csv("smsspamcollection",sep="\t",names=["label","message"])# tab separated file
df.head()

lm = WordNetLemmatizer()
corpus = []  # to take thefinal list of lamatized data

for i in range(0, len(df)):
    data = re.sub("^[a-zA-Z]", " ", df["message"][i])  # 1
    data = data.lower()  # 2
    data = data.split()  # 3
    data = [lm.lemmatize(words) for words in data if not words in stopwords.words("english")]
    data = " ".join(data)  # join it to make a corpus
    corpus.append(data)

from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer(max_features=5000) #taking only 5000 frequent words
X=tf.fit_transform(corpus).toarray()
y=pd.get_dummies(df["label"],drop_first=True)

#splitting the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import MultinomialNB
mb=MultinomialNB()
mb.fit(X_train,y_train)
y_pred=mb.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,y_pred))

print("accuracy",accuracy_score(y_test,y_pred))

pickle.dump(tf,open("tfidf.pkl","wb"))
pickle.dump(mb,open("model.pkl","wb"))




