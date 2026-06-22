import streamlit as st
from langchain_groq import ChatGroq

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        api_key=st.secrets["GROQ_API_KEY"]
    )