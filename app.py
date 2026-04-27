import streamlit as st
import pickle
import re
import spacy

# Page config (must be first Streamlit command)
st.set_page_config(page_title="Airline Sentiment Analysis")

# Cache spaCy model (VERY IMPORTANT for deployment)
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])

nlp = load_spacy_model()

# Load ML components
@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
    return model, vectorizer, le

model, vectorizer, le = load_models()

# Text cleaning function (same as your training logic)
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
    ]

    return " ".join(tokens)

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

        st.write("**Processed Text:**", cleaned)

        if label == "positive":
            st.success("Positive 😊")
        elif label == "negative":
            st.error("Negative 😡")
        else:
            st.info("Neutral 😐")
