# pages/1_Tutor_Bot.py
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from utils.vectorstore import load_vectorstore
from utils.llm import load_llm
from utils.student_profile import get_weak_topics, init_db
from utils.diagram_renderer import render_response_with_diagrams

# ── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tutor Bot — Smart Tutor",
    page_icon="📚",
    layout="centered"
)

init_db()

# ── GUARD: must be logged in ───────────────────────────────────────────────
if not st.session_state.get("student_id"):
    st.warning("Please log in from the Home page first.")
    st.page_link("Home.py", label="Go to Home →")
    st.stop()

student_id   = st.session_state.student_id
student_name = st.session_state.student_name

st.title("📚 Curriculum Tutor Bot")
st.caption(f"Personalised for {student_name} — answers only from your curriculum")

# ── LOAD MODELS ────────────────────────────────────────────────────────────
vectorstore = load_vectorstore()
retriever   = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 15
    }
)

# ── GET WEAK TOPICS FOR PERSONALISATION ───────────────────────────────────
weak_topics = get_weak_topics(student_id)
weak_str    = ""
if weak_topics:
    weak_str = "\n\nSTUDENT WEAK TOPICS (spend extra time here):\n" + \
               "\n".join(f"- {t['subject']}: {t['topic']}"
                         for t in weak_topics)

# ── RAG ANSWER FUNCTION ────────────────────────────────────────────────────
def get_answer(user_input, chat_history, retriever):

    retrieved_docs = retriever.invoke(user_input)

    # Debug
    st.sidebar.write("Retrieved docs:", len(retrieved_docs))
    st.sidebar.write("Question:", user_input)

    for doc in retrieved_docs[:5]:
        st.sidebar.write(
            f"{doc.metadata.get('filename')} | Page {doc.metadata.get('page')}"
        )

    if not retrieved_docs or all(
        len(doc.page_content.strip()) < 20
        for doc in retrieved_docs
    ):
        return (
            "I could not find relevant content in your PDFs "
            "for this question.",
            []
        )

    context = "\n\n".join(
        f"[From: {doc.metadata.get('filename','unknown')} "
        f"Page {doc.metadata.get('page','?')}]\n"
        f"{doc.page_content}"
        for doc in retrieved_docs
    )

   

    sources = [
        f"{doc.metadata.get('filename','?')} "
        f"p.{doc.metadata.get('page','?')}"
        for doc in retrieved_docs[:3]
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a curriculum tutor for {student_name}.

You ONLY answer using the CONTEXT provided below.

STRICT RULES:
1. Use ONLY the information contained in the CONTEXT. 
   You may rephrase, explain, simplify, and solve questions using the information from the CONTEXT.

   Do not introduce facts that are not supported by the CONTEXT.
2. If the user asks a question that appears in the context
   (including exercise questions),
   - Answer naturally as a teacher.
   - Do not say "the context says" or "the answer is mentioned in the context".
   - Explain the concept directly.
   - Mention the source at the end.

3. If the user asks an exercise question,
   - solve every part asked
   - show step-by-step working
   - provide the final answer clearly
4. Do not say the question was not provided if it appears in the context.

5. Mention the file/page used.
6. If the context does NOT contain the answer — say:
   "This topic is not in the retrieved sections."
7. You can ask the student if he wants to learn more about related topics, but do not force it.
8. After answering, ask one short checking question
9. For exercise-solving questions, do NOT ask a checking question unless the student asks for practice.

{weak_str}

CONTEXT:
─────────────────────────────────────────
{{context}}
─────────────────────────────────────────
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    llm = load_llm()

    formatted = prompt.format_messages(
        context=context,
        chat_history=chat_history,
        input=user_input
    )
    # DEBUG
    st.sidebar.write("Context length:", len(context))
    st.sidebar.write("Context preview:")
    st.sidebar.code(context[:2000])

    st.sidebar.write("Messages sent to LLM:")
    st.sidebar.code(str(formatted)[:3000])

    response = llm.invoke(formatted)

    return response.content, sources

# ── SESSION STATE ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"**Student:** {student_name}")
    st.markdown(f"**Class:** {st.session_state.student_class}")
    if weak_topics:
        st.markdown("**⚠️ Weak topics:**")
        for t in weak_topics[:3]:
            st.markdown(f"- {t['topic']}")
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# ── DISPLAY CHAT ───────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            f"Hello {student_name}! I'm your curriculum tutor. "
            "Ask me anything about your subjects!"
        )

# ── CHAT INPUT ─────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask your question here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = get_answer(
                user_input,
                st.session_state.chat_history,
                retriever
            )
        st.markdown(answer)
        #render_response_with_diagrams(answer)
        if sources:
            with st.expander("📄 Sources used"):
                for s in sources:
                    st.write(f"• {s}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=answer))