#assessment_bot_backup
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

from utils.vectorstore import load_vectorstore
from utils.llm import load_llm

from utils.student_profile import (
    init_db,
    get_weak_topics,
    update_topic_score,
    save_assessment_attempt,
    get_assessment_history,
    save_question_log,
    get_or_create_student,
    get_student_class,
    get_topics_for_class_subject
)

# ---------------------------------------------------
# PAGE
# ---------------------------------------------------

st.set_page_config(
    page_title="Assessment Bot",
    page_icon="📝",
    layout="centered"
)

init_db()

# ---------------------------------------------------
# LOGIN GUARD
# ---------------------------------------------------

if not st.session_state.get("student_id"):
    st.warning("Please log in first.")
    st.page_link("Home.py", label="Go to Home")
    st.stop()

student_id = st.session_state.student_id
student_name = st.session_state.student_name

st.title("📝 Assessment Bot")

with st.expander("📊 Assessment History"):

    history = get_assessment_history(
        student_id
    )

    if history:

        for h in history:

            st.write(
                f"{h['created_at'][:10]} | "
                f"{h['subject']} | "
                f"{h['topic']} | "
                f"{round(h['score']*100,1)}%"
            )

    else:
            st.rerun()
