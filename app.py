import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Load trained model and TF-IDF vectorizer
with open("vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Text preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for word in text:
        if word.isalnum():
            y.append(word)

    filtered_words = []
    stop_words = set(stopwords.words("english"))

    for word in y:
        if word not in stop_words and word not in string.punctuation:
            filtered_words.append(word)

    return " ".join(filtered_words)


# Streamlit interface
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)

st.title("📩 Email / SMS Spam Classifier")
st.write("Enter a message below to check whether it is spam or legitimate.")

input_sms = st.text_area(
    "Enter your message",
    placeholder="Type or paste your message here...",
    height=150
)

if st.button("Predict Message", type="primary"):

    if input_sms.strip() == "":
        st.warning("Please enter a message first.")

    else:
        # 1. Preprocess the input text
        transformed_sms = transform_text(input_sms)

        # 2. Convert text into TF-IDF features
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict using the trained model
        result = model.predict(vector_input)[0]

        # 4. Display the result
        if result == 1:
            st.error("🚨 Spam Detected!")
            st.write("This message may be unwanted or fraudulent.")

        else:
            st.success("✅ Not Spam (Ham)")
            st.write("This message is classified as legitimate.")