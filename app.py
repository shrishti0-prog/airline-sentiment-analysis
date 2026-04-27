import streamlit as st
import pickle
import re
import spacy

st.set_page_config(page_title="Sentiment Analysis")

# load spacy
nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

# load files
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))


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


st.title("✈ Airline Tweet Sentiment Analysis")

tweet = st.text_area("Enter Tweet")

if st.button("Predict"):

    cleaned = clean_text(tweet)

    vec = vectorizer.transform([cleaned])

    pred = model.predict(vec)

    label = le.inverse_transform(pred)[0]

    if label == "positive":
        st.success("Positive 😊")

    elif label == "negative":
        st.error("Negative 😡")

    else:
        st.info("Neutral 😐")
