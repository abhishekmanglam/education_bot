import os
import streamlit as st
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

def get_project_root():
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "Home.py").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent

@st.cache_resource
def load_embeddings():
    # Set HF token
    hf_token = st.secrets.get("HF_TOKEN", None)
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token

    # Cache model locally in project folder
    cache_dir = str(get_project_root() / "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=cache_dir
    )