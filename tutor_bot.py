# tutor_bot.py — Streamlit Cloud compatible
# Uses HuggingFace embeddings + Groq LLM (both free, no Ollama needed)

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
#from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
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
# Dynamic paths — work on both your laptop AND Streamlit Cloud
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "pdfs")
#CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
FAISS_DIR = os.path.join(BASE_DIR, "FAISS_db")

# ── LOAD EMBEDDINGS ────────────────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    """
    HuggingFace embeddings — free, runs directly in Python.
    No Ollama, no API key, no server needed.
    Downloads the model once (~90MB) on first run.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ── LOAD LLM ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_llm():
    """
    Groq — free cloud LLM API.
    Reads API key from Streamlit Secrets.
    """
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=st.secrets["GROQ_API_KEY"]
    )

# ── LOAD VECTORSTORE ───────────────────────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    """
    Loads PDFs → chunks → embeds → stores in Chroma.
    Runs only once per session due to @st.cache_resource.
    """
    #embeddings = load_embeddings()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # ── DEBUG: show what Streamlit Cloud can see ───────────────────────
    st.sidebar.markdown("### 📁 File Check")
    st.sidebar.write(f"Looking for PDFs in:")
    st.sidebar.code(PDF_FOLDER)
    st.sidebar.write(f"Folder exists: {os.path.exists(PDF_FOLDER)}")

    if os.path.exists(PDF_FOLDER):
        found_files = [f for f in os.listdir(PDF_FOLDER)
                       if f.endswith(".pdf")]
        st.sidebar.write(f"PDFs found: {len(found_files)}")
        for f in found_files:
            st.sidebar.write(f"  • {f}")
    # ── END DEBUG ──────────────────────────────────────────────────────

    # Load all PDFs
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
                st.sidebar.write(f"✓ Loaded: {filename} "
                                 f"({len(pages)} pages)")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Failed: {filename} — {e}")

    # No PDFs found
    if not all_documents:
        st.error(
            "❌ No PDFs found. "
            "Make sure your PDFs are in the /pdfs folder "
            "in your GitHub repository."
        )
        st.stop()

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_documents)
    st.sidebar.success(f"✓ {len(chunks)} chunks ready")

    # Embed and store
    # vectorstore = Chroma.from_documents(
    #     documents=chunks,
    #     embedding=embeddings,
    #     persist_directory=CHROMA_DIR
    # )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


# ── RAG ANSWER FUNCTION ────────────────────────────────────────────────────
def get_answer(user_input: str,
               chat_history: list,
               retriever) -> tuple[str, list]:

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(user_input)

    # Guard: empty context
    if not retrieved_docs or all(
        len(doc.page_content.strip()) < 20
        for doc in retrieved_docs
    ):
        return (
            "I could not find relevant content in your PDFs "
            "for this question. Try rephrasing or ask about "
            "a topic covered in the loaded documents.",
            []
        )

    # Format context with source labels
    context = "\n\n".join(
        f"[From: {doc.metadata.get('filename', 'unknown')} "
        f"Page {doc.metadata.get('page', '?')}]\n"
        f"{doc.page_content}"
        for doc in retrieved_docs
    )

    # Source list for display
    sources = list(set([
        f"{doc.metadata.get('filename', '?')} "
        f"p.{doc.metadata.get('page', '?')}"
        for doc in retrieved_docs
    ]))

    # Strict prompt — no hallucination allowed
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a curriculum tutor.
You ONLY answer using the CONTEXT provided below.

STRICT RULES:
1. Use ONLY the CONTEXT — never your general knowledge
2. If the answer is in the context — answer directly and
   mention which file/page it came from
3. If the context does NOT contain the answer — say exactly:
   "This topic is not in the retrieved sections.
    Try asking about a related topic."
4. Never say "I don't have access to files" —
   the content IS in the CONTEXT below
5. After answering, ask one short checking question

CONTEXT:
─────────────────────────────────────────
{context}
─────────────────────────────────────────"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    llm = load_llm()

    formatted = prompt.format_messages(
        context=context,
        chat_history=chat_history,
        input=user_input
    )

    response = llm.invoke(formatted)
    return response.content, sources


# ── INITIALISE SESSION STATE ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── LOAD MODELS ────────────────────────────────────────────────────────────
vectorstore = load_vectorstore()
retriever   = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# ── DISPLAY CHAT HISTORY ───────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── WELCOME MESSAGE ────────────────────────────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Hello! I'm your curriculum tutor. "
            "Ask me anything about your loaded subjects!"
        )

# ── CHAT INPUT ─────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask your question here..."):

    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = get_answer(
                user_input,
                st.session_state.chat_history,
                retriever
            )

        st.markdown(answer)

        if sources:
            with st.expander("📄 Sources used"):
                for source in sources:
                    st.write(f"• {source}")

    # Save to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
    st.session_state.chat_history.append(
        HumanMessage(content=user_input)
    )
    st.session_state.chat_history.append(
        AIMessage(content=answer)
    )