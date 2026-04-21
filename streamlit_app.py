import streamlit as st
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob
import re

st.title("Input text below for sentiment analysis")
input_text = st.text_area("Enter a sentence", key="input")
submit_button = st.button("Analyze")


#Keeping only Text and digits
input_text = re.sub(r"[^A-Za-z0-9]", " ", input_text)
#Removes Whitespaces
input_text = re.sub(r"\'s", " ", input_text)
# Removing Links if any
input_text = re.sub(r"http\S+", " link ", input_text)
# Removes Punctuations and Numbers
input_text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", input_text)
# Splitting Text
input_text = input_text.split()
# Lemmatizer
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in input_text]
input_text = " ".join(lemmatized_words)

blob = TextBlob(text=input_text)
sentiment_score = blob.sentiment.polarity

if submit_button:
    st.write(input_text + " has sentiment score " + str(sentiment_score))

