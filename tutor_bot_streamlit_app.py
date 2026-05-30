# tutor_bot.py — Streamlit Cloud compatible version

import warnings
warnings.filterwarnings("ignore")

import os
import json
import streamlit as st

# ✅ Updated imports — all work on Streamlit Cloud
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Curriculum Tutoring Bot",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Curriculum Tutoring Bot")
st.caption("Powered by Llama 3 + RAG — answers only from your curriculum")

# ── PATHS ──────────────────────────────────────────────────────────────────
PDF_FOLDER = "./pdfs"
# On Streamlit Cloud this means the pdfs/ folder
# inside your GitHub repository

CHROMA_DIR = "./chroma_db"
# Chroma will build here on first run

# ── LLM AND EMBEDDINGS ─────────────────────────────────────────────────────
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama3-8b-8192",
        temperature=0.3,
        api_key=st.secrets["GROQ_API_KEY"]
    )

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        # Downloads once, cached for the session
    )

# ── VECTORSTORE ────────────────────────────────────────────────────────────
@st.cache_resource
def load_vectorstore():

    embeddings = load_embeddings()

    # Load from disk if already built
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        st.sidebar.success("✓ Loaded vectorstore from disk")
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )

    # Build fresh from PDFs
    st.sidebar.info("⏳ Building vectorstore from PDFs...")

    all_documents = []

    if os.path.exists(PDF_FOLDER):
        pdf_files = [f for f in os.listdir(PDF_FOLDER)
                     if f.endswith(".pdf")]

        for filename in pdf_files:
            path = os.path.join(PDF_FOLDER, filename)
            try:
                loader = PyPDFLoader(path)
                pages = loader.load()
                for page in pages:
                    page.metadata["filename"] = filename
                    page.metadata["subject"] = \
                        filename.replace(".pdf", "")
                all_documents.extend(pages)
                st.sidebar.write(f"✓ Loaded: {filename}")
            except Exception as e:
                st.sidebar.warning(f"Failed: {filename} — {e}")

    if not all_documents:
        st.error(
            "No PDFs found in ./pdfs folder. "
            "Add PDFs to your GitHub repo under /pdfs/"
        )
        st.stop()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_documents)

    # Embed and store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    st.sidebar.success(f"✓ Indexed {len(chunks)} chunks")
    return vectorstore


# ── REST OF YOUR CODE STAYS THE SAME ──────────────────────────────────────
# load models
llm        = load_llm()
vectorstore = load_vectorstore()
retriever  = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# ... your get_answer(), sidebar, chat loop etc unchanged