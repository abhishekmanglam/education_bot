# app.py — Streamlit web UI for your tutoring bot

import warnings
warnings.filterwarnings("ignore")

import os
import json
import streamlit as st
# st is the Streamlit object — everything UI-related goes through it
# st.title(), st.chat_message(), st.chat_input() etc.

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "pdfs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Curriculum Tutoring Bot",
    page_icon="📚",
    layout="centered"
)
# Must be the first Streamlit call in the file
# Sets the browser tab title and icon

st.title("📚 Curriculum Tutoring Bot")
st.caption("Powered by Llama 3.2 + RAG — answers only from your curriculum")


# ── LOAD VECTORSTORE (cached so it only runs once) ────────────────────────
@st.cache_resource
# @st.cache_resource is critical —
# without it, the vectorstore reloads on EVERY message
# with it, it loads once and stays in memory for the whole session
def load_vectorstore():
    """
    Load PDFs → chunk → embed → return vectorstore.
    Runs only once per session due to @st.cache_resource.
    """

    # PDF_FOLDER = r"C:\Users\abhis\OneDrive\Documents\GitHub\education_bot\pdfs" 
    # CHROMA_DIR = r"C:\Users\abhis\OneDrive\Documents\GitHub\education_bot\app_chroma_db"
    PDF_FOLDER = os.path.join(BASE_DIR, "pdfs")
    CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

    # If vectorstore already exists on disk — load it directly
    # No need to re-embed every time you restart the app
    if os.path.exists(CHROMA_DIR):
        st.sidebar.success("✓ Loaded existing vectorstore from disk")
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=OllamaEmbeddings(model="nomic-embed-text")
        )

    # Otherwise — build it fresh from PDFs
    st.sidebar.info("Building vectorstore from PDFs...")

    all_documents = []

    if os.path.exists(PDF_FOLDER):
        pdf_files = [f for f in os.listdir(PDF_FOLDER)
                     if f.endswith(".pdf")]

        for filename in pdf_files:
            file_path = os.path.join(PDF_FOLDER, filename)
            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                for page in pages:
                    page.metadata["filename"] = filename
                    page.metadata["subject"] = filename.replace(".pdf","")
                all_documents.extend(pages)
            except Exception as e:
                st.sidebar.warning(f"Could not load {filename}: {e}")

    # Fallback to sample text if no PDFs found
    if not all_documents:
        st.sidebar.warning("No PDFs found — using sample curriculum")
        sample = """
        Photosynthesis converts sunlight into food using chlorophyll.
        The equation: 6CO2 + 6H2O + light → C6H12O6 + 6O2
        Mitosis produces two identical daughter cells.
        Phases: prophase, metaphase, anaphase, telophase.
        """
        with open("sample.txt", "w") as f:
            f.write(sample)
        loader = TextLoader("sample.txt", encoding="utf-8")
        all_documents = loader.load()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_documents)

    # Embed and store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()

    st.sidebar.success(f"✓ Indexed {len(chunks)} chunks from your PDFs")
    return vectorstore


# ── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Settings")

    # Show which PDFs are loaded
    #PDF_FOLDER = r"C:\Users\abhis\OneDrive\Documents\GitHub\education_bot\pdfs"
    PDF_FOLDER = os.path.join(BASE_DIR, "pdfs")
    if os.path.exists(PDF_FOLDER):
        pdf_files = [f for f in os.listdir(PDF_FOLDER)
                     if f.endswith(".pdf")]
        if pdf_files:
            st.subheader("Loaded PDFs:")
            for f in pdf_files:
                st.write(f"• {f}")
        else:
            st.warning("No PDFs in ./pdfs folder")
    
    # Slider to control how many chunks are retrieved
    k_value = st.slider(
        "Chunks to retrieve per question",
        min_value=1,
        max_value=6,
        value=3,
        help="More chunks = more context but slower response"
    )
    # st.slider creates an interactive slider in the sidebar
    # User can drag it to change k_value at runtime

    # Button to clear chat history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()
        # st.rerun() refreshes the whole page
        # clearing the displayed messages


# ── INITIALISE SESSION STATE ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
# st.session_state persists data across reruns
# Streamlit reruns the entire script on every user interaction
# Without session_state, chat history would reset on every message
# messages = list of {"role": "assistant"/"user", "content": "..."}
# Used to display the chat bubbles

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# chat_history = list of HumanMessage/AIMessage objects
# Passed to the LLM so it remembers previous turns
# Separate from messages (which is just for display)


# ── LOAD MODELS ───────────────────────────────────────────────────────────
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": k_value
        
    }
)
llm = ChatOllama(model="llama3.2", temperature=0.3)


# ── RAG FUNCTION ──────────────────────────────────────────────────────────
def get_answer(user_input: str, chat_history: list) -> tuple[str, list]:
    """
    Same RAG logic as test_bot.py —
    retrieve → format → prompt → LLM → return answer + sources
    """

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(user_input)


    # ── DEBUG: print what was actually retrieved ──────────────────────────
    print(f"\n{'='*60}")
    print(f"QUERY: {user_input}")
    print(f"RETRIEVED {len(retrieved_docs)} chunks:")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n  Chunk {i+1}:")
        print(f"  File: {doc.metadata.get('filename', 'unknown')}")
        print(f"  Page: {doc.metadata.get('page', '?')}")
        print(f"  Text preview: '{doc.page_content[:150]}...'")
        print(f"  Text length: {len(doc.page_content)} chars")
    print(f"{'='*60}\n")
    # ── END DEBUG ─────────────────────────────────────────────────────────

    # Check if context is actually empty
    if not retrieved_docs or all(
        len(doc.page_content.strip()) < 20
        for doc in retrieved_docs
    ):
        return (
            "I couldn't find relevant content in your PDFs for this question. "
            "This might mean the PDF text wasn't extracted correctly, "
            "or this topic isn't covered in the loaded documents.",
            []
        )


    # Format chunks into context string
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Build source list for display
    sources = list(set([
        f"{doc.metadata.get('filename', 'curriculum')} "
        f"p.{doc.metadata.get('page', '?')}"
        for doc in retrieved_docs
    ]))
    # set() removes duplicates — same file/page might appear twice

    # Build prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a curriculum tutor. You have been given
excerpts from the student's study material as CONTEXT below.

STRICT RULES — follow these without exception:
1. Answer ONLY using information from the CONTEXT provided
2. Do NOT use your general training knowledge under any circumstances
3. Do NOT say things like "based on my training" or "I believe"
4. If the answer is clearly present in the CONTEXT — answer it directly
   and cite which part of the context you used
5. If the CONTEXT is empty or contains no relevant information — say
   exactly this: "I could not find this topic in your loaded PDFs.
   Please check that the PDF containing this topic was loaded correctly."
6. Never pretend you don't have context when context IS provided below
7. After answering, ask one checking question to verify understanding

CONTEXT FROM YOUR PDFS:
─────────────────────────────
{context}
─────────────────────────────
If the above context contains relevant information, use it.
If it is empty, say so honestly."""),

        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # Format and invoke
    formatted = prompt.format_messages(
        context=context,
        chat_history=chat_history,
        input=user_input
    )
    response = llm.invoke(formatted)

    return response.content, sources


# ── DISPLAY CHAT HISTORY ──────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # st.chat_message creates a chat bubble
        # "user" = right-aligned human bubble
        # "assistant" = left-aligned bot bubble
        st.markdown(message["content"])
        # st.markdown renders text with formatting


# ── WELCOME MESSAGE ───────────────────────────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Hello! I'm your curriculum tutor. "
            "Ask me anything about the subjects in your PDFs. "
            "What would you like to learn today?"
        )


# ── CHAT INPUT ────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask your question here..."):
    # st.chat_input creates the text box at the bottom
    # := is the walrus operator — assigns AND checks in one line
    # The if block only runs when the user submits a message

    # Show user message immediately
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and show bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # st.spinner shows a loading animation while waiting
            answer, sources = get_answer(
                user_input,
                st.session_state.chat_history
            )

        st.markdown(answer)

        # Show sources in a collapsible section
        if sources:
            with st.expander("📄 Sources used"):
                # st.expander creates a collapsible section
                # Click to expand and see which chunks were used
                for source in sources:
                    st.write(f"• {source}")

    # Save to session state for next turn
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