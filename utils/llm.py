import streamlit as st
from langchain_groq import ChatGroq

@st.cache_resource
def load_llm():

    key = st.secrets["GROQ_API_KEY"]

    st.write("Key exists:", bool(key))
    st.write("Key starts with:", key[:10])

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        api_key=key
    )
