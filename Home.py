import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from utils.student_profile import init_db, get_or_create_student, \
    get_all_scores, get_weak_topics

# ── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Tutor",
    page_icon="🎓",
    layout="centered"
)

# ── INIT DATABASE ──────────────────────────────────────────────────────────
init_db()

# ── SESSION STATE ──────────────────────────────────────────────────────────
if "student_id" not in st.session_state:
    st.session_state.student_id = None
if "student_name" not in st.session_state:
    st.session_state.student_name = None
if "student_class" not in st.session_state:
    st.session_state.student_class = None

# ── HEADER ─────────────────────────────────────────────────────────────────
st.title("🎓 Smart Tutor")
st.markdown("*Your personalised AI learning companion*")
st.divider()

# ── LOGIN FORM ─────────────────────────────────────────────────────────────
if st.session_state.student_id is None:

    st.subheader("👋 Welcome! Let's get started")
    st.markdown("Please enter your details to begin.")

    with st.form("login_form"):
        name   = st.text_input("Your Name", placeholder="e.g. Arjun Sharma")
        class_ = st.selectbox(
            "Your Class",
            ["Class 7", "Class 8", "Class 9", "Class 10"]
        )
        submit = st.form_submit_button("Start Learning 🚀")

    if submit:
        if not name.strip():
            st.error("Please enter your name.")
        else:
            student_id = get_or_create_student(name.strip(), class_)
            st.session_state.student_id    = student_id
            st.session_state.student_name  = name.strip()
            st.session_state.student_class = class_
            st.rerun()

# ── DASHBOARD (after login) ────────────────────────────────────────────────
else:
    name   = st.session_state.student_name
    class_ = st.session_state.student_class

    st.subheader(f"Welcome back, {name}! 👋")
    st.markdown(f"**Class:** {class_}")
    st.divider()

    # Navigation cards
    st.subheader("📚 What would you like to do?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📖 Tutor Bot")
        st.markdown("Ask questions about your curriculum. "
                    "Get explanations with source references.")
        st.switch_page("Pages/1_tutor_bot.py", label="Open Tutor Bot →")

    with col2:
        st.markdown("### 📝 Assessment Bot")
        st.markdown("Test your knowledge. "
                    "Adaptive questions based on your weak areas.")
        st.switch_page("Pages/2_Assessment_bot.py",
                     label="Open Assessment Bot →")

    with col3:
        st.markdown("### 📋 Content Delivery")
        st.markdown("Structured lessons delivered "
                    "topic by topic from your syllabus.")
        st.switch_page("Pages/3_Content_Delivery.py",
                     label="Open Content Bot →")

    st.divider()

    # Progress dashboard
    st.subheader("📊 Your Progress")

    all_scores   = get_all_scores(st.session_state.student_id)
    weak_topics  = get_weak_topics(st.session_state.student_id)

    if not all_scores:
        st.info("No assessment data yet. "
                "Complete some assessments to see your progress!")
    else:
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Topics Attempted", len(all_scores))
        with col_b:
            avg = sum(s["score"] for s in all_scores) / len(all_scores)
            st.metric("Average Score", f"{avg*100:.0f}%")
        with col_c:
            st.metric("Weak Topics", len(weak_topics))

        if weak_topics:
            st.markdown("#### ⚠️ Topics needing attention:")
            for t in weak_topics[:5]:
                st.markdown(
                    f"- **{t['subject'].title()}** — {t['topic']} "
                    f"({t['score']*100:.0f}%)"
                )

    st.divider()

    # Logout
    if st.button("🚪 Switch Student"):
        st.session_state.student_id    = None
        st.session_state.student_name  = None
        st.session_state.student_class = None
        st.rerun()
