# Import Library
import re
import random
import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string

# Memuat Model dan Parameter yang Telah Disimpan
model_path = 'model.h5'  # Sesuaikan dengan path model Anda
tokenizer_path = 'tokenizer.pkl'
max_seq_length_path = 'max_sequence_length.pkl'
label_encoder_path = 'le.pkl'
responses_path = 'responses.pkl'

chatbot_model = load_model(model_path)  # Memuat model
tokenizer = pickle.load(open(tokenizer_path, 'rb'))  # Memuat tokenizer
max_sequence_length = pickle.load(open(max_seq_length_path, 'rb'))  # Memuat panjang maksimum urutan
label_encoder = pickle.load(open(label_encoder_path, 'rb'))  # Memuat label encoder
responses = pickle.load(open(responses_path, 'rb'))  # Memuat respons bot

# Aplikasi Streamlit
st.title("Mental Health ChatBot")
st.write("Say Hi to MentalHealth ChatBot")

# Menyimpan histori chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Menampilkan histori
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input pengguna
prompt = st.chat_input("Type your chat here")

if prompt:
    # Preprocessing input pengguna
    prediction_input = [letters.lower() for letters in prompt if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p = [prediction_input]
    
    # Tokenisasi dan padding
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input], maxlen=max_sequence_length)
    
    # Prediksi tag
    output = chatbot_model.predict(prediction_input)
    output = output.argmax()
    
    # Mendapatkan respons berdasarkan tag
    response_tag = label_encoder.inverse_transform([output])[0]
    bot_response = random.choice(responses[response_tag]) if response_tag in responses else "I'm not sure I understand."
    
    # Menampilkan chat
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        st.write(bot_response)
    
    # Menyimpan histori chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
