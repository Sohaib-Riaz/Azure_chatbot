import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load settings from .env
load_dotenv()

# Connect to the deployed Azure model
client = OpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    base_url=os.getenv("AZURE_BASE_URL")
)

MODEL_NAME = os.getenv("AZURE_MODEL")

st.title("My Azure AI Chatbot")
st.caption("Powered by my deployed GPT model")

# Store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box
prompt = st.chat_input("Type your message...")

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = client.responses.create(
                    model=MODEL_NAME,
                    input=st.session_state.messages
                )

                answer = response.output_text
                st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            st.error(f"API Error: {e}")