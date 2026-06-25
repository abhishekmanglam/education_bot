import streamlit as st

VALID_USERS = {
    "Abhishek": "pass1234",
    "Aliah": "pass1234",
    "Anshu": "pass1234",
    "Tasneem": "pass1234",
    "Kirti": "pass1234",
    "Swara": "pass1234",
    "Aparajita": "pass1234",
    "Aditya": "pass1234",
    "Linesh": "pass1234",
    "Kewal": "pass1234"
    
}

def login():
    if st.session_state.get("authenticated"):
        return True

    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if VALID_USERS.get(username) == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password")

    return False
