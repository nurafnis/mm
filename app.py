import streamlit as st
import numpy as np
import pickle
import random
import string
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model dan objek yang diperlukan
chatbot_model = load_model('model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
max_sequence_length = pickle.load(open('max_sequence_length.pkl', 'rb'))
le = pickle.load(open('le.pkl', 'rb'))
responses = pickle.load(open('responses.pkl', 'rb'))

# Header aplikasi
st.title("🤖 Chatbot")
st.markdown("Berbicara dengan Chatbot yang cerdas! Ketik pesanmu di bawah.")

# Input pengguna
user_input = st.text_input("👨‍🦰 Kamu:", "")

if st.button("Kirim") and user_input:
    # Preprocessing input
    prediction_input = [letters.lower() for letters in user_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    
    # Tokenisasi dan padding
    prediction_input = tokenizer.texts_to_sequences([prediction_input])
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input], max_sequence_length)

    # Prediksi respons
    output = chatbot_model.predict(prediction_input)
    output = output.argmax()

    # Temukan tag respons
    # Mendapatkan respons bot dengan validasi
if response_tag in responses and len(responses[response_tag]) > 0:
    bot_response = random.choice(responses[response_tag])
else:
    bot_response = "I'm sorry, I didn't understand that."
    
    # Tampilkan hasil
    st.text_area("🤖 Chatbot:", bot_response, height=100)

    # Periksa jika pengguna ingin keluar
    if response_tag == "goodbye":
        st.stop()
