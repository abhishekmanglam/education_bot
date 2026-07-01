import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.embeddings import load_embeddings
from pathlib import Path


def get_project_root():
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "Home.py").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent


BASE_DIR   = get_project_root()
PDF_FOLDER = str(BASE_DIR / "pdfs")
FAISS_DIR  = str(BASE_DIR / "faiss_db")



@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()

   
    index_file = os.path.join(FAISS_DIR, "index.faiss")
    
    if os.path.exists(index_file):
        
        st.sidebar.success("✓ Loading existing FAISS index")
        return FAISS.load_local(
            FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # index_file = os.path.join(FAISS_DIR, "index.faiss")

    # # ---------- DEBUG ----------
    # st.sidebar.write("BASE_DIR:", BASE_DIR)
    # st.sidebar.write("FAISS_DIR:", FAISS_DIR)
    # st.sidebar.write("Looking for:", index_file)

    # if os.path.exists(BASE_DIR):
    #     st.sidebar.write("BASE_DIR contents:")
    #     st.sidebar.write(os.listdir(BASE_DIR))

    # if os.path.exists(FAISS_DIR):
    #     st.sidebar.write("faiss_db contents:")
    #     st.sidebar.write(os.listdir(FAISS_DIR))
    # else:
    #     st.sidebar.write("faiss_db directory DOES NOT EXIST")

    # st.sidebar.write("index.faiss exists:", os.path.exists(index_file))
    # # ---------------------------

    # if os.path.exists(index_file):

    #     st.sidebar.success("✓ Loading existing FAISS index")
    #     return FAISS.load_local(
    #         FAISS_DIR,
    #         embeddings,
    #         allow_dangerous_deserialization=True
    #     )

    st.sidebar.info("No FAISS index found. Building one...")
    return build_vectorstore(embeddings)


def build_vectorstore(embeddings):
    """Build FAISS index from PDFs and save to disk."""
    st.sidebar.markdown("### 📁 Building index from PDFs...")
    st.sidebar.write("Checking PDF files...")
    st.sidebar.write(f"BASE_DIR: {BASE_DIR}")
    st.sidebar.write(f"PDF_FOLDER: {PDF_FOLDER}")
    st.sidebar.write(f"Folder exists: {os.path.exists(PDF_FOLDER)}")
    
    all_documents = []

    if not os.path.exists(PDF_FOLDER):
        st.error("❌ No pdfs/ folder found.")
        st.stop()
    
    st.sidebar.write("Folder contents:")
    st.sidebar.write(os.listdir(PDF_FOLDER))

    pdf_files = []

    for root, dirs, files in os.walk(PDF_FOLDER):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    st.sidebar.write(f"Found {len(pdf_files)} PDF files")
    st.sidebar.write(pdf_files[:5])

    if not pdf_files:
        st.error("❌ No PDFs found in pdfs/ folder.")
        st.stop()

    # Step 1: Load PDFs
    st.sidebar.info("Step 1: Loading PDFs...")
    for path in pdf_files:
        
        filename = os.path.basename(path)
        st.sidebar.write(f"Loading {filename}")
    
        try:
            
            loader = PyPDFLoader(path)
            pages  = loader.load()
            for page in pages:
                page.metadata["filename"] = filename
                name = os.path.splitext(filename)[0]
                parts = name.split("_")
                page.metadata["subject"] = parts[0] if parts else name
                page.metadata["class"]   = parts[1] if len(parts) > 1 else "unknown"
                page.metadata["topic"] = parts[2] if len(parts) > 2 else "unknown"

            all_documents.extend(pages)
            st.sidebar.write(f"✓ {filename} ({len(pages)} pages)")
        except Exception as e:
            st.sidebar.error(f"❌ Failed loading {filename}: {e}")

    if not all_documents:
        st.error("❌ Could not load any PDF content.")
        st.stop()

    st.sidebar.success(f"Step 1 done: {len(all_documents)} pages loaded")

    # Step 2: Chunk
    st.sidebar.info("Step 2: Chunking...")
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(all_documents)
        st.sidebar.success(f"Step 2 done: {len(chunks)} chunks created")
    except Exception as e:
        st.sidebar.error(f"❌ Chunking failed: {e}")
        st.stop()

    # Step 3: Embed
    st.sidebar.info("Step 3: Embedding — this takes a few minutes first time...")
    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.sidebar.success("Step 3 done: Embedding complete")
    except Exception as e:
        st.sidebar.error(f"❌ Embedding failed: {e}")
        import traceback
        st.sidebar.code(traceback.format_exc())
        st.stop()

    # Step 4: Save
    st.sidebar.info("Step 4: Saving to disk...")
    try:
        os.makedirs(FAISS_DIR, exist_ok=True)
        vectorstore.save_local(FAISS_DIR)
        saved = os.path.exists(os.path.join(FAISS_DIR, "index.faiss"))
        if saved:
            st.sidebar.success("Step 4 done: ✓ FAISS index saved!")
        else:
            st.sidebar.error("Step 4: Save ran but file not found")
    except Exception as e:
        st.sidebar.error(f"❌ Save failed: {e}")
        import traceback
        st.sidebar.code(traceback.format_exc())

    return vectorstore
