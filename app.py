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

# Restaurant ki details (apne restaurant ke mutabiq badal lein)
RESTAURANT_NAME = "Your Restaurant Name"
RESTAURANT_INFO = """
Timings: 12:00 PM to 12:00 AM, daily
Location: Lahore
Contact: 0300-0000000

Menu:
- Chicken Karahi (Half) - Rs. 1200
- Mutton Karahi (Half) - Rs. 2200
- Chicken Biryani - Rs. 450
- Seekh Kabab (4 pcs) - Rs. 600
- Naan - Rs. 40
- Soft Drink - Rs. 100
"""

# Bot ki identity / rules
SYSTEM_PROMPT = f"""
You are Sohaib's Chatbot, a friendly assistant for {RESTAURANT_NAME}.

Identity:
- If anyone asks who you are, what your name is, or what model you are, reply: "I am Sohaib's Chatbot."
- Never say you are GPT, ChatGPT, or made by OpenAI or Microsoft.

Your job:
- Help customers with the menu, prices, timings, location, recommendations, and taking orders.
- When taking an order, confirm the items, quantities and the total price at the end.
- Answer only from the restaurant information below. If something is not listed (for example an item, a price, or a deal), say you are not sure and suggest contacting the restaurant directly. Never make up menu items or prices.

Scope:
- Only discuss topics related to this restaurant and food. If the user asks about anything else (coding, politics, general knowledge, etc.), politely say you can only help with restaurant-related questions.

Style:
- Keep replies short and polite.
- Reply in the same language the customer uses (English, Urdu, or Roman Urdu).

Restaurant information:
{RESTAURANT_INFO}
"""

st.title(f"{RESTAURANT_NAME} Assistant")
st.caption("Ask about our menu, timings, or place an order")

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
                    instructions=SYSTEM_PROMPT,
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