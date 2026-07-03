# ==========================================================
# Airline Sentiment Analysis using Word2Vec + ANN
# Developed by: Your Name
# ==========================================================

import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import re

from gensim.models import Word2Vec

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="Airline Sentiment Analyzer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# Load Models
# ----------------------------------------------------------

@st.cache_resource
def load_models():

    model = tf.keras.models.load_model(
        "airline_sentiment_word2vec.keras"
    )

    word2vec = Word2Vec.load(
        "word2vec.model"
    )

    scaler = pickle.load(
        open("scaler.pkl","rb")
    )

    encoder = pickle.load(
        open("label_encoder.pkl","rb")
    )

    return model, word2vec, scaler, encoder


model, word2vec, scaler, encoder = load_models()


# ----------------------------------------------------------
# Text Cleaning
# ----------------------------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+"," ",text)

    text = re.sub(r"@\w+"," ",text)

    text = re.sub(r"[^a-zA-Z ]"," ",text)

    text = re.sub(r"\s+"," ",text).strip()

    return text.split()


# ----------------------------------------------------------
# Sentence Embedding
# ----------------------------------------------------------

def sentence_vector(sentence):

    words = clean_text(sentence)

    vectors = []

    for word in words:

        if word in word2vec.wv:

            vectors.append(
                word2vec.wv[word]
            )

    if len(vectors)==0:

        return np.zeros(200)

    return np.mean(vectors,axis=0)


# ----------------------------------------------------------
# Prediction Function
# ----------------------------------------------------------

def predict_sentiment(text):

    vector = sentence_vector(text)

    vector = vector.reshape(1,-1)

    vector = scaler.transform(vector)

    prediction = model.predict(
        vector,
        verbose=0
    )

    pred_class = np.argmax(prediction)

    sentiment = encoder.inverse_transform(
        [pred_class]
    )[0]

    confidence = np.max(prediction)

    return sentiment, confidence, prediction[0]


# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------

with st.sidebar:

    st.title("✈ Airline Sentiment")

    st.markdown("---")

    st.write("### About")

    st.info(
        """
This application predicts the sentiment of airline tweets using:

✅ Word2Vec

✅ Artificial Neural Network (ANN)

✅ TensorFlow/Keras

✅ Streamlit
"""
    )

    st.markdown("---")

    st.write("### Model")

    st.success("Word2Vec + ANN")

    st.markdown("---")

    st.write("### Classes")

    st.write("😊 Positive")

    st.write("😐 Neutral")

    st.write("😠 Negative")
# ==========================================================
# Custom CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background:#0E1117;
}

.big-title{
    text-align:center;
    color:#4FC3F7;
    font-size:45px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:white;
    font-size:20px;
}

.result-card{
    background:#1B263B;
    padding:25px;
    border-radius:15px;
    border:2px solid #4FC3F7;
}

.footer{
    text-align:center;
    color:gray;
    font-size:15px;
}

</style>
""",unsafe_allow_html=True)


# ==========================================================
# Hero Section
# ==========================================================

st.markdown(
"""
<div class='big-title'>
✈ Airline Sentiment Analyzer
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class='sub-title'>
Predict whether an airline tweet is
<b>Positive</b>,
<b>Neutral</b>,
or
<b>Negative</b>
using
<b>Word2Vec + Artificial Neural Network</b>.
</div>
""",
unsafe_allow_html=True
)

st.write("")
st.write("")


# ==========================================================
# Text Input
# ==========================================================

tweet = st.text_area(
    "📝 Enter a Tweet",
    height=180,
    placeholder="Example: I had an amazing flight. The crew was very friendly and helpful!"
)

col1,col2 = st.columns([1,1])

with col1:
    predict_btn = st.button(
        "🚀 Predict Sentiment",
        use_container_width=True
    )

with col2:
    clear_btn = st.button(
        "🗑 Clear",
        use_container_width=True
    )

if clear_btn:
    st.rerun()


# ==========================================================
# Sample Tweets
# ==========================================================

st.markdown("### 💡 Sample Tweets")

sample1,sample2,sample3 = st.columns(3)

with sample1:
    st.info("The flight was fantastic and the crew was amazing!")

with sample2:
    st.info("The flight was okay. Nothing special.")

with sample3:
    st.info("Worst airline ever. My baggage is missing.")


# ==========================================================
# Prediction
# ==========================================================

if predict_btn:

    if tweet.strip()=="":

        st.warning("Please enter a tweet.")

    else:

        sentiment,confidence,probabilities = predict_sentiment(tweet)

        if sentiment.lower()=="positive":

            emoji="😊"
            color="green"

        elif sentiment.lower()=="negative":

            emoji="😠"
            color="red"

        else:

            emoji="😐"
            color="orange"


        st.write("")
        st.write("")


        st.markdown(
        f"""
        <div class="result-card">

        <h2 style="color:{color}; text-align:center;">

        {emoji} {sentiment.upper()}

        </h2>

        <h4 style="text-align:center; color:white;">

        Confidence :
        {confidence*100:.2f}%

        </h4>

        </div>

        """,
        unsafe_allow_html=True
        )


        st.write("")
        st.subheader("📊 Prediction Probabilities")

        labels = encoder.classes_

        for label,prob in zip(labels,probabilities):

            st.write(f"**{label.capitalize()}**")

            st.progress(float(prob))

            st.write(f"{prob*100:.2f}%")

            st.write("")
# ==========================================================
# Session Prediction History
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if predict_btn and tweet.strip() != "":

    st.session_state.history.insert(
        0,
        {
            "Tweet": tweet,
            "Prediction": sentiment,
            "Confidence": f"{confidence*100:.2f}%"
        }
    )


st.markdown("---")

st.subheader("📜 Prediction History")

if len(st.session_state.history) == 0:

    st.info("No predictions made yet.")

else:

    for i,item in enumerate(st.session_state.history):

        with st.expander(f"Prediction {i+1}"):

            st.write("**Tweet**")

            st.write(item["Tweet"])

            st.write("**Prediction**")

            st.success(item["Prediction"])

            st.write("**Confidence**")

            st.write(item["Confidence"])
st.markdown("---")

st.subheader("📊 Model Information")

col1,col2,col3 = st.columns(3)

with col1:

    st.metric(
        "Model",
        "ANN"
    )

with col2:

    st.metric(
        "Embedding",
        "Word2Vec"
    )

with col3:

    st.metric(
        "Classes",
        "3"
    )
st.markdown("---")

st.subheader("📚 About This Project")

st.write("""
This application predicts the sentiment of airline-related tweets.

### Workflow

✔ Text Cleaning

✔ Word2Vec Embedding

✔ Sentence Vector Creation

✔ Standard Scaling

✔ Artificial Neural Network

✔ Sentiment Prediction

The model classifies tweets into:

- 😊 Positive

- 😐 Neutral

- 😠 Negative
""")
st.markdown("---")

st.markdown(
"""
<div class="footer">

Made with ❤️ using

<b>Streamlit</b> |
<b>TensorFlow</b> |
<b>Word2Vec</b>

</div>
""",
unsafe_allow_html=True
)