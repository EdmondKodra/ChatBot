import json
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

intents = json.load(open('intents.json'))

tags = []
patterns = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

vector = TfidfVectorizer()
patterns_scaled = vector.fit_transform(patterns)

Bot = LogisticRegression(max_iter=100000)
Bot.fit(patterns_scaled,tags)

def ChatBot(input_message):
    input_message = vector.transform([input_message])
    pred_tag = Bot.predict(input_message)[0]
    for intent in intents['intents']:
        if intent['tag'] == pred_tag:
            return random.choice(intent['responses'])
    return "?"

st.title("ChatBot Cars")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):
    
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"AI ChatBot:" + ChatBot(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    

    st.session_state.messages.append({"role": "assistant", "content": response})
