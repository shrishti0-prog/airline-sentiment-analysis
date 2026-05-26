import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Airline Sentiment Analysis",
    page_icon="✈",
    layout="centered"
)

# ---------------- LOAD STOPWORDS ---------------- #

@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return set(stopwords.words('english'))

stop_words = load_stopwords()

# ---------------- LOAD MODEL ---------------- #

@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
    return model, vectorizer, le

model, vectorizer, le = load_models()

# ---------------- TEXT CLEANING ---------------- #

def clean_text(text):

    text = re.sub(r"http\S+", "", text)

    text = text.lower()

    text = re.sub(r"[^a-z\s]", "", text)

    text = re.sub(r"\s+", " ", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("📌 About Project")

st.sidebar.info(
    """
    This application predicts airline tweet sentiment using:

    ✅ Natural Language Processing (NLP)

    ✅ TF-IDF Vectorization

    ✅ Machine Learning

    Sentiments:
    • Positive 😊
    • Negative 😠
    • Neutral 😐
    """
)

# ---------------- MAIN TITLE ---------------- #

st.title("✈ Airline Tweet Sentiment Analysis")

st.markdown(
    """
    Analyze airline-related tweets and classify their sentiment using Machine Learning and NLP.
    """
)

# ---------------- ACCURACY ---------------- #

st.success("✅ Model Accuracy: 89%")

# ---------------- EXAMPLE TWEETS ---------------- #

st.markdown("### 💡 Example Tweets")

st.code(
    """
The flight was amazing
Worst airline ever
The service was okay
Flight delayed badly
Staff behavior was excellent
"""
)

# ---------------- TEXT AREA ---------------- #

tweet = st.text_area(
    "📝 Enter Tweet",
    height=150,
    placeholder="Type your airline tweet here..."
)

# ---------------- PREDICTION ---------------- #

if st.button("🔍 Predict Sentiment"):

    if tweet.strip() == "":

        st.warning("⚠ Please enter some text")

    else:

        with st.spinner("Analyzing sentiment..."):

            # preprocessing
            cleaned = clean_text(tweet)

            # vectorization
            vec = vectorizer.transform([cleaned])

            # prediction
            pred = model.predict(vec)

            label = le.inverse_transform(pred)[0]

        # processed text
        st.markdown("### 🧹 Processed Text")

        st.info(cleaned)

        # prediction output
        st.markdown("### 📊 Sentiment Prediction")

        if label == "positive":

            st.success("😊 Positive Sentiment")

            st.balloons()

        elif label == "negative":

            st.error("😠 Negative Sentiment")

        else:

            st.warning("😐 Neutral Sentiment")

# ---------------- FOOTER ---------------- #

st.markdown("---")

st.caption(
    "Developed using Python, NLP, TF-IDF, Machine Learning, and Streamlit"
)
