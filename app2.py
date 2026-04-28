import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Page config (must be first)
st.set_page_config(page_title="Airline Sentiment Analysis")

# Download stopwords (cached so it runs once)
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return set(stopwords.words('english'))

stop_words = load_stopwords()

# Load ML components
@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
    return model, vectorizer, le

model, vectorizer, le = load_models()

# Text cleaning (matches NLTK-based training)
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# UI
st.title("✈ Airline Tweet Sentiment Analysis")

tweet = st.text_area("Enter Tweet")

if st.button("Predict"):
    if tweet.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(tweet)

        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)
        label = le.inverse_transform(pred)[0]

        st.write("Processed Text:", cleaned)

        if label == "positive":
            st.success("Positive 😊")
        elif label == "negative":
            st.error("Negative 😡")
        else:
            st.info("Neutral 😐")
