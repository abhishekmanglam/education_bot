import streamlit as st
from langchain_groq import ChatGroq

@st.cache_resource
def load_llm(temperature=0.4):

    # st.write("Secrets keys:", list(st.secrets.keys()))

    if "GROQ_API_KEY" in st.secrets:
        key = st.secrets["GROQ_API_KEY"]
    #     st.write("Key prefix:", key[:10])
    else:
        st.error("GROQ_API_KEY NOT FOUND")
        st.stop()

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=key
    )
