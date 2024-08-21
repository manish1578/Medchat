import streamlit as st
import time
import pandas as pd
import numpy as np
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import joblib
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Custom CSS for styling
custom_css = """
<style>
    .stApp {
        background: linear-gradient(135deg, #7b4397, #dc2430);
        color: #FFFFFF;
        font-family: 'Arial', sans-serif;
        padding: 40px;  /* Increased padding */
    }
    .stTitle {
        text-align: left;
        color: #FFFFFF;
        font-size: 50px;  /* Increased font size for the title */
        font-weight: bold;
    }
    .stSubtitle {
        font-size: 25px;  /* Font size for the subtitle */
        color: #FFFFFF;
    }
    .stSelectbox label, .stTextInput label {
        color: #FFFFFF;
        font-size: 30px;  /* Increased font size */
    }
    .stTextInput input, .stSelectbox select {
        background-color: #FFFFFF;
        color: #002D62;
        font-size: 30px;  /* Increased font size */
        padding: 15px;  /* Increased padding */
        text-align: left;  /* Align text input to the left */
    }
    .stButton button {
        background-color: #00BCD4;
        color: #FFFFFF;
        font-size: 30px;  /* Increased font size */
        padding: 15px;  /* Increased padding */
    }
    .custom-container {
        display: flex;
        justify-content: flex-start; /* Align items to the left */
        align-items: flex-start;
    }
    .custom-container .left {
        flex: 1;
        padding: 20px;
        font-size: 30px;  /* Increased font size */
    }
    .custom-container .right {
        flex: 0.5;
        padding: 20px;
    }
    .custom-container .right img {
        width: 100%;
        height: auto;
    }
    .stImage img {
        border-radius: 15px;
    }
    .chat-window {
        background-color: black; /* Black background for the chat window */
        padding: 20px;
        border-radius: 15px;
        color: white; /* White text for better readability */
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# File paths
#voting_classifier_path = r"C:\Users\khwai\Pluto-Healthcare-Chatbot-main\Pluto-Healthcare-Chatbot-main\Pickle Files\voting_classifier.pkl"
#label_encoder_path = r"C:\Users\khwai\Pluto-Healthcare-Chatbot-main\Pluto-Healthcare-Chatbot-main\Pickle Files\label_encoder.pkl"
#tfidf_vectorizer_path = r"C:\Users\khwai\Pluto-Healthcare-Chatbot-main\Pluto-Healthcare-Chatbot-main\Pickle Files\tfidf_vectorizer.pkl"
#df_medicine_path = r"C:\Users\khwai\Pluto-Healthcare-Chatbot-main\Pluto-Healthcare-Chatbot-main\Datasets\Medicine_Details.csv"
#df_details_path = r"C:\Users\khwai\Pluto-Healthcare-Chatbot-main\Pluto-Healthcare-Chatbot-main\Datasets\Disease_Description.csv"
#nn_model_path = r"C:\Users\khwai\Pluto-Healthcare-Chatbot-main\Pluto-Healthcare-Chatbot-main\Pickle Files\nn_model.pkl"


voting_classifier_path = r"\voting_classifier.pkl"
label_encoder_path = r"\label_encoder.pkl"
tfidf_vectorizer_path = r"\tfidf_vectorizer.pkl"
df_medicine_path = r"\Datasets\Medicine_Details.csv"
df_details_path = r"\Datasets\Disease_Description.csv"
nn_model_path = r"\nn_model.pkl"


# Load models and data
def load_pickle_file(filepath):
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {filepath}")
        return None

voting_classifier = load_pickle_file(voting_classifier_path)
label_encoder = load_pickle_file(label_encoder_path)
tfidf_vectorizer = load_pickle_file(tfidf_vectorizer_path)

df_medicine = pd.read_csv(df_medicine_path)
df_details = pd.read_csv(df_details_path)

def preprocess_text(text):
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens if token.lower() not in stop_words]
    return ' '.join(tokens)

def append_user(prompt):
    with st.chat_message("User"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "User", "content": prompt})

def append_assistant(prompt):
    with st.chat_message("Assistant"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "Assistant", "content": prompt})

def bot_response(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.07)

def append_bot(response):
    with st.chat_message("Assistant"):
        st.write_stream(bot_response(response))
    st.session_state.messages.append({"role": "Assistant", "content": response})

def append_med_bot(response_df):
    response_str = response_df.to_markdown()
    with st.chat_message("Assistant"):
        st.markdown(response_str)
        time.sleep(0.07)
    st.session_state.messages.append({"role": "Assistant", "content": response_str})

def predict_disease(text):
    text_processed = preprocess_text(text)
    st.write(f"Processed text: {text_processed}")  # Debugging: check processed text
    if tfidf_vectorizer is None:
        st.error("TF-IDF Vectorizer not loaded. Please check the file path or ensure it is fitted.")
        return
    text_transformed = tfidf_vectorizer.transform([text_processed])
    st.write(f"Transformed text shape: {text_transformed.shape}")  # Debugging: check transformed text shape
    predicted_label_encoded = voting_classifier.predict(text_transformed)
    predicted_label = label_encoder.inverse_transform(predicted_label_encoded)
    append_bot("You are probably suffering from: " + predicted_label[0])
    row = df_details[df_details['Disease'] == predicted_label[0]]
    if not row.empty:
        description = row['Description'].values[0]
        symp = row['Symptoms'].values[0]
        treatment = row['Treatment'].values[0]
        append_bot("**About the disease:** \n" + description)
        append_bot("**Symptoms:** \n" + symp)
        append_bot("**Treatment:** \n" + treatment)

def get_wikipedia_page_url(topic):
    try:
        search_results = wikipedia.search(topic)
        append_bot("Please choose one of the following results:")
        selected_result = st.selectbox("Choose a search result:", search_results[:5], index=0)
        append_user(selected_result)
        page = wikipedia.page(selected_result)
        return page.url
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Ambiguous search term. Did you mean: {', '.join(e.options)}?"
    except wikipedia.exceptions.PageError:
        return "Page not found on Wikipedia."

def load_models(vectorizer_path, model_path):
    try:
        tfidf_vectorizer = joblib.load(vectorizer_path)
        nn_model = joblib.load(model_path)
        return tfidf_vectorizer, nn_model
    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}")
        return None, None

def recommend_medicines_by_text(input_text, tfidf_vectorizer, nn_model, df_medicine):
    input_vector = tfidf_vectorizer.transform([input_text])
    st.write(f"Input vector shape: {input_vector.shape}")  # Debugging: check input vector shape
    if input_vector.shape[1] == 0:
        st.error("Input vector has 0 features. Check the TF-IDF vectorizer and the input text.")
        return pd.DataFrame()
    distances, indices = nn_model.kneighbors(input_vector)
    recommended_medicines_df = df_medicine.iloc[indices[0]][['Medicine Name', 'Composition', 'Uses', 'Side_effects']]
    return recommended_medicines_df

def main():
    st.markdown(custom_css, unsafe_allow_html=True)

    # Insert an image at the top left
    st.image(r"C:\Users\khwai\Pluto-Healthcare-Chatbot-main\Pluto-Healthcare-Chatbot-main\Main file\logo.png", width=140)  # Adjust the width as needed

    st.title("MEDCHAT 🩺 : Your Health One Chat Away💬🏥💊🌟")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_topic" not in st.session_state:
        st.session_state.selected_topic = ""
    if "section_headings" not in st.session_state:
        st.session_state.section_headings = None

    # Only one column for better left alignment
    col1 = st.container()

    with col1:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"],unsafe_allow_html=True)

    # Only one column for better left alignment
    col1 = st.container()

    with col1:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)
        st.markdown('<div class="chat-window"><div class="stSubtitle">Hello, health seekers! 🌿 I’m here to support you on your journey to better health.</div>', unsafe_allow_html=True)

        action = st.selectbox("Choose an option:", [
                              "Choose ", "Monitor Symptoms", "Uncover details about various health conditions"], index=0)

        if action == "Monitor Symptoms":
            append_user(action)
            if symptoms := st.chat_input("Enter your symptoms:"):
                append_assistant("Processing your symptoms...")
                append_user(symptoms)
                predict_disease(symptoms)
            else:
                st.write("Assistant: Please provide your symptoms")

        elif action == "Uncover details about various health conditions":
            append_user(action)
            with st.form(key='disease_info_form'):
                topic = st.text_input("Enter the disease name:")
                submit_button = st.form_submit_button("Submit")
                if submit_button and topic:
                    append_user(topic)
                    url = get_wikipedia_page_url(topic)
                    if "Did you mean:" in url:
                        append_bot(url)
                    else:
                        append_bot("Here's a Wikipedia page about " + topic + ": " + url)
                elif submit_button:
                    st.write("Assistant: Please provide a disease name")

if __name__ == "__main__":
    main()
