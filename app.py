import streamlit as st
from keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and variables
chatbot_model = load_model('/content/model.h5')
tokenizer = pickle.load(open('/content/tokenizer.pkl','rb'))
max_sequence_length = pickle.load(open('/content/max_sequence_length.pkl','rb'))
le = pickle.load(open('/content/le.pkl','rb'))
responses = pickle.load(open('/content/responses.pkl','rb'))

# Streamlit app
st.title("ChatbotX")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Apa yang ingin Anda tanyakan?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Preprocess user input
    texts_p = []
    prediction_input = prompt
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input], max_sequence_length)

    # Get model prediction
    output = chatbot_model.predict(prediction_input)
    output = output.argmax()
    response_tag = le.inverse_transform([output])[0]
    response = random.choice(responses[response_tag])

    # Add bot message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Display bot message in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
