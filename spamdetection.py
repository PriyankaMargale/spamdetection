import streamlit as st
import pickle
import string
import nltk
import google.generativeai as genai
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# Load model/vectorizer from Google Drive
# import pickle
# vectorizer = pickle.load(open("/content/drive/MyDrive/spam_modelvectorizer.pkl", 'rb'))
# model = pickle.load(open("/content/drive/MyDrive/spam_modelmodel.pkl", 'rb'))

# UI
import google.generativeai as genai




genai.configure(api_key="AIzaSyCTrkF10TMtLapEmaNaGj8KTp92kUnd-jM")
st.title("Multi-Lingual Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")
st.write("OR")
flag = False
uploaded_file = st.file_uploader("Upload a Txt file...")
if uploaded_file is not None:
  with open("demofile.txt", "w") as f:
    f.write(uploaded_file.read().decode())
  flag = True
import re
import string

def transform_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    # Remove digits
    text = re.sub(r"\d+", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

if st.button('Predict'):
    gemini = genai.GenerativeModel('gemini-1.5-flash')
    # response = model.generate_content("What is the meaning of life in few words?")
    if flag:
      input_sms = open("demofile.txt").read()
      flag = False

    # Preprocess
      # transformed_sms = transform_text(input_sms)
      # vector_input = vectorizer.transform([transformed_sms])

      # # Predict
      # result = model.predict(vector_input)[0]
    prompt1 = f"Classify whether the Email Content is Spam or Not Spam. Content could in any language Understand the Language and classify the email Content.\n Email Content: {input_sms}\n\nNote: response should be 'spam' or 'Not spam'"
    
    response1 = gemini.generate_content(prompt1)
    # Vectorize
    
    label = "Spam" if response1.text.lower() == "Spam".lower() else "Not Spam"
    st.header(f"Prediction: {response1.text}")

    # Gemini explanation
    with st.spinner("explanation..."):
        prompt = f"The following message was classified as '{label}'. Explain why this message may be considered as such:\n\n{input_sms}"
        try:
            response = gemini.generate_content(prompt)
            
            explanation = response.text
            st.markdown("### Prediction Explanation")
            st.info(explanation)
        except Exception as e:
            st.error("Gemini failed to respond. Check your API key or quota.")
