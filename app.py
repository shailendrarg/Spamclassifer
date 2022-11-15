import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import pickle
import streamlit as st
import re #regular expression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lm=WordNetLemmatizer()
# for lamatisation which creates root words

#unpickle the vectorizer and model

tf = pickle.load(open("tfidf.pkl","rb"))
model=pickle.load(open("model.pkl","rb"))
st.title("SMS Spam classifier")

message =st.text_area("Please enter the message for prediction")
if st.button("Predict"):

    def clean_message(message):
        data = re.sub("[^a-zA-Z]", " ",message)  # 1
        data = data.lower()  # 2
        data = data.split()  # 3
        data = [lm.lemmatize(words) for words in data if not words in stopwords.words("english")]
        data = " ".join(data)  # join it to make a corpus
        return data
    transform_input=clean_message(message)
    print("transform input",transform_input)
    model_input=tf.transform([transform_input])
    print("model input",model_input)
    result=model.predict(model_input)[0]
    print("result",result)
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")
