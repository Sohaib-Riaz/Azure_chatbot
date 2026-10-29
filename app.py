
import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from rag import MenuRAG

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
api_key = os.getenv("AZURE_API_KEY")
base_url = os.getenv("AZURE_BASE_URL")
MODEL_NAME = os.getenv("AZURE_MODEL")

if not api_key or not base_url or not MODEL_NAME:
    st.error(
        "Azure configuration missing. Please check "
        "AZURE_API_KEY, AZURE_BASE_URL and AZURE_MODEL in .env."
    )
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

# Restaurant details
RESTAURANT_NAME = "Desi Dastarkhwan"

RESTAURANT_INFO = """
Timings: 12:00 PM to 12:00 AM, daily
Location: Lake City Raiwind
Contact: 0300-8850051
"""

# Initialize PDF search and embeddings once
@st.cache_resource
def load_rag():
    return MenuRAG(
        "Desi_Restaurant_Menu.pdf",
        use_dense=True
    )

# Page setup
st.title(f"{RESTAURANT_NAME} Assistant")
st.caption(
    "Ask about our menu, ingredients, prices, "
    "vegetarian dishes, or place an order."
)

# Load menu PDF and embedding model
try:
    rag = load_rag()
except Exception as e:
    st.error(f"Could not load the menu PDF or search system: {e}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input("Type your message...")

if prompt:
    # Display and save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching menu and preparing answer..."):

                # Step 1: Search the PDF for relevant information
                hits = rag.search(prompt, k=5)

                # Step 2: Build the retrieved menu context
                if hits:
                    menu_context = rag.build_context(hits)
                else:
                    menu_context = (
                        "No relevant menu information was retrieved. "
                        "Do not guess menu items, ingredients, prices, "
                        "or vegetarian status."
                    )

                # Step 3: Create instructions for Azure OpenAI
                system_prompt = f"""
You are the friendly restaurant assistant for
{RESTAURANT_NAME}.

IDENTITY:
- If asked your name, say: "I am Sohaib's Chatbot."
- Never claim to be GPT, ChatGPT, or made by OpenAI or Microsoft.

RULES:
- Answer restaurant-related questions only.
- Use the retrieved MENU CONTEXT for menu items, ingredients,
  prices, vegetarian status, and other menu details.
- Never invent dishes, ingredients, prices, discounts, or deals.
- If the menu context does not provide the answer, say that
  the menu does not mention the requested information.
- Do not assume that missing information means something is
  definitely unavailable or absent.
- Use RESTAURANT INFORMATION for opening times, location,
  and contact details.
- For orders, confirm items and quantities with the customer.
- Calculate an order total only when the prices are available.
- Do not claim an order has been placed or paid for.
- Reply in the language used by the customer: English, Urdu,
  or Roman Urdu.
- Keep replies short, clear, and polite.
- For unrelated questions, explain that you only assist
  with restaurant-related queries.

RESTAURANT INFORMATION:
{RESTAURANT_INFO}

MENU CONTEXT RETRIEVED FROM PDF:
{menu_context}
"""

                # Step 4: Send conversation to Azure OpenAI
                response = client.responses.create(
                    model=MODEL_NAME,
                    instructions=system_prompt,
                    input=st.session_state.messages
                )

                answer = response.output_text

                if not answer:
                    answer = (
                        "Sorry, I could not generate a response. "
                        "Please try again."
                    )

                # Display answer
                st.markdown(answer)

                # Display sources used for retrieval
                if hits:
                    with st.expander("Menu sources used"):
                        for chunk, score in hits:
                            st.write(f"- {chunk.name}")

            # Save assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            st.error(f"Error: {e}")
