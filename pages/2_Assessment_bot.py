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

        st.write(
            "No assessments attempted yet."
        )



# ---------------------------------------------------
# LOAD VECTORSTORE
# ---------------------------------------------------

vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 30
    }
)

llm = load_llm()

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

defaults = {
    "assessment_started": False,
    "assessment_complete": False,
    "question_number": 0,
    "score": 0,
    "current_question": "",
    "current_context": "",
    "current_subject": "",
    "current_topic": "",
    "difficulty": "Medium",
    "current_difficulty": "Medium",
    "total_questions": 5,
    "asked_questions": [],
    "question_history": [],
    "final_score": None,
    "current_sources": "",
    "show_feedback": False,
    "feedback_text": "",
    "last_score": 0,
    "current_answer": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state.assessment_complete:

    st.success(
        f"""
Assessment Complete!

Final Score:
{st.session_state.final_score}%

Subject:
{st.session_state.current_subject}

Topic:
{st.session_state.current_topic}
"""
    )

    if st.button("Start New Assessment"):

        st.session_state.assessment_complete = False
        st.session_state.final_score = None

        st.rerun()

    st.stop()

# ---------------------------------------------------
# TOPIC SELECTION
# ---------------------------------------------------

if not st.session_state.assessment_started:



    st.subheader("Assessment Setup")

    mode = st.radio(
        "Choose Assessment Mode",
        [
            "Curriculum Topic", "Custom Topic",
            "Surprise me (weak topic)"
        ]
    )

    subject = st.selectbox(
        "Subject",
        [
            "Maths",
            "Science",
            "English",
            "Social Science"
        ]
    )

    if mode == "Curriculum Topic":

        # topic = st.text_input(
        #     "Topic",
        #     placeholder="e.g. Motion, Triangles, Force"
        # )

        student_class = get_student_class(student_id)

        topics = get_topics_for_class_subject(vectorstore,student_class,subject)

        topic = st.selectbox("Topic",topics)
        
    elif mode == "custom_topic":
        custom_topic = st.text_input("Enter topic",placeholder="Fractions")

    else:

        weak_topics = get_weak_topics(
            student_id,
            subject=subject
        )

        if weak_topics:

            weakest = weak_topics[0]

            topic = weakest["topic"]

            st.info(
                f"We selected your weakest "
                f"{subject} topic: {topic}"
            )

        else:

            topic = "Random"

            st.info(
                f"No weak topics found for "
                f"{subject}. Random assessment selected."
            )

    difficulty = st.selectbox(
        "Difficulty",
        [
            "Easy",
            "Medium",
            "Hard",
            "Adaptive"
        ]
    )

    total_questions = st.slider(
        "Number of Questions",
        3,
        10,
        5
    )

    if st.button("Start Assessment"):

        st.session_state.assessment_started = True
        st.session_state.current_subject = subject
        st.session_state.current_topic = topic
        st.session_state.difficulty = difficulty

        st.session_state.current_difficulty = (
            "Medium"
            if difficulty == "Adaptive"
                else difficulty
        )
        st.session_state.total_questions = total_questions

        st.session_state.question_number = 0
        st.session_state.score = 0
        st.session_state.asked_questions = []
        st.session_state.question_history = []

        st.rerun()

# ---------------------------------------------------
# QUESTION GENERATION
# ---------------------------------------------------

def generate_question(subject, topic, difficulty):

    question_types = [
    "Mixed",
    "MCQ",
    "True/False",
    "Fill in the Blanks",
    "Short Answer",
    "Numerical",
    "Application",
    "Reasoning",
    "Higher Order Thinking"
]

    question_type = question_types[
        st.session_state.question_number %
        len(question_types)
]
    import random
    if topic == "Random":

        random_queries = [
            f"{subject} chapter",
            f"{subject} exercise",
            f"{subject} concept",
            f"{subject} numerical",
            f"{subject} theorem"
        ]

        query = random.choice(random_queries)

    else:
        query = f"{subject} {topic}"

    docs = retriever.invoke(query)

    st.sidebar.write(
    "Current Difficulty:",
    difficulty
)

    st.sidebar.write(
    f"Retrieved {len(docs)} docs for question "
    f"{st.session_state.question_number + 1}"
    )

    for d in docs[:4]:
        st.sidebar.write(
            d.metadata.get("filename"),
            d.metadata.get("page")
        )

    if len(docs) > 6:
        docs = random.sample(docs, 6)

    context = "\n\n".join(
    doc.page_content
    for doc in docs
    )

    sources = []

    for doc in docs:

        filename = doc.metadata.get(
            "filename",
            "Unknown File"
        )

        page = doc.metadata.get(
            "page",
            "Unknown Page"
        )

        sources.append(
            f"{filename} | Page {page}"
        )

    prompt = ChatPromptTemplate.from_template("""
You are an assessment generator.

Topic:
{topic}

Difficulty:
{difficulty}

Difficulty Rules:

Easy:
- direct recall
- definitions
- simple examples

Medium:
- application of concepts
- multi-step thinking

Hard:
- HOTS questions
- reasoning
- case-based
- difficult numericals

Previously asked questions:
{previous_questions}

Context:
{context}
                                              
Question Type:
{question_type} questions
from the retrieved context.

Rules:
1. Generate ONE new question.
2. NEVER repeat any question listed above.
3. NEVER ask the same concept using different wording.
4. Choose a different part of the context whenever possible.
5. If previous questions are from exercises,
   select another exercise.
6. Do not provide the answer.
7. Output ONLY the question.
8. Avoid starting the question with the same words
   as any previous question.
9. Generate:

- Use only the context.
- For MCQ provide 4 options.
- For True/False provide statement only.
- For Fill in the Blanks provide blanks.
- For Short Answer ask conceptual questions.

40% MCQ
20% True/False
20% Fill in the Blanks
20% Subjective.
""")

    response = llm.invoke(
    prompt.format_messages(
        difficulty=difficulty,
        topic=topic,
        context=context[:5000],
        question_type=question_type,
        previous_questions="\n".join(
            st.session_state.asked_questions
        )
    )
)
    new_question = response.content.strip()

    if new_question.lower() in [
        q.lower()
        for q in st.session_state.asked_questions
    ]:

        retry_prompt = ChatPromptTemplate.from_template("""
    Generate a completely different question.

    Topic:
    {topic}

    Previous Questions:
    {previous_questions}

    Rules:
    1. Must be different from every question listed.
    2. Must test another concept.
    3. Output only the question.
    """)

        retry_response = llm.invoke(
            retry_prompt.format_messages(
                topic=topic,
                previous_questions="\n".join(
                    st.session_state.asked_questions
                )
            )
        )

        new_question = retry_response.content.strip()

    return (
        new_question,
        context,
        "\n".join(sources)
    )
  


# ---------------------------------------------------
# ANSWER EVALUATION
# ---------------------------------------------------

def evaluate_answer(question, answer, context,sources):

    prompt = ChatPromptTemplate.from_template(
        """
You are a teacher.

Question:
{question}

Student Answer:
{answer}

Reference Context:
{context}

Sources:
{sources}

Grade the answer.

Return ONLY:

Return ONLY in this format:

Score: X

Feedback:
<short feedback>

Correct Answer:
<model answer>

Explanation:
<step-by-step explanation based on context>

Sources Used:
<relevant source names>

Where score is between 0 and 1
"""
    )

    result = llm.invoke(
        prompt.format_messages(
            question=question,
            answer=answer,
            context=context[:5000],
            sources=sources
        )
    )

    text = result.content

    score = 0

    try:

        score_line = text.split("Score:")[1]
        score = float(
            score_line.split("\n")[0].strip()
        )

    except:

        score = 0

    return score, text

# ---------------------------------------------------
# ASSESSMENT FLOW
# ---------------------------------------------------

if st.session_state.assessment_started:

    st.info(
        f"Current Difficulty: "
        f"{st.session_state.current_difficulty}"
    )

    with st.expander("Assessment Progress"):

        for i, item in enumerate(
            st.session_state.question_history,
            start=1
        ):

            st.write(
                f"Question {i}: "
                f"{round(item['score'] * 100)}%"
            )

    current_avg = (
        st.session_state.score
        /
        max(1, st.session_state.question_number)
    )

    st.write(
        f"Current Average: "
        f"{round(current_avg * 100, 1)}%"
    )

    st.write(
        f"### Question "
        f"{st.session_state.question_number + 1}"
        f"/{st.session_state.total_questions}"
    )

    # ----------------------------------------
    # GENERATE QUESTION
    # ----------------------------------------

    if st.session_state.current_question == "":

        q, ctx, src = generate_question(
            st.session_state.current_subject,
            st.session_state.current_topic,
            st.session_state.current_difficulty
        )

        st.session_state.asked_questions.append(q)

        st.session_state.current_question = q
        st.session_state.current_context = ctx
        st.session_state.current_sources = src

    st.markdown(
        st.session_state.current_question
    )

    # ----------------------------------------
    # SHOW FEEDBACK SCREEN
    # ----------------------------------------

    if st.session_state.show_feedback:

        st.markdown("## Evaluation")
        (
            st.session_state.feedback_text
        )

        # from utils.diagram_renderer import (
        #     render_response_with_diagrams
        # )

        # render_response_with_diagrams(
        #     st.session_state.feedback_text
        # )

        if st.button("Next Question"):

            score = st.session_state.last_score

            # Save score
            st.session_state.score += score

            st.session_state.question_number += 1

            st.session_state.question_history.append(
                {
                    "question":
                    st.session_state.current_question,

                    "score":
                    score
                }
            )

            # -------------------------
            # Adaptive Difficulty
            # -------------------------

            if (
                st.session_state.difficulty
                == "Adaptive"
            ):

                if score >= 0.8:

                    st.session_state.current_difficulty = "Hard"

                elif score >= 0.5:

                    st.session_state.current_difficulty = "Medium"

                else:

                    st.session_state.current_difficulty = "Easy"

            # -------------------------
            # FINISH ASSESSMENT
            # -------------------------

            if (
                st.session_state.question_number
                >=
                st.session_state.total_questions
            ):

                final_score = (
                    st.session_state.score
                    /
                    st.session_state.total_questions
                )

                update_topic_score(
                    student_id,
                    st.session_state.current_subject,
                    st.session_state.current_topic,
                    final_score
                )

                save_assessment_attempt(
                    student_id,
                    st.session_state.current_subject,
                    st.session_state.current_topic,
                    final_score,
                    st.session_state.total_questions,
                    st.session_state.current_difficulty
                )

                st.session_state.final_score = round(
                    final_score * 100,
                    1
                )

                st.session_state.assessment_complete = True

                st.session_state.assessment_started = False

            # Prepare next question

            st.session_state.current_question = ""

            st.session_state.current_answer = ""

            st.session_state.show_feedback = False

            st.session_state.feedback_text = ""

            st.rerun()

    # ----------------------------------------
    # ANSWER ENTRY
    # ----------------------------------------

    else:

        answer = st.text_area(
            "Your Answer",
            key="current_answer"
        )

        if st.button("Submit Answer"):

            score, evaluation = evaluate_answer(
                st.session_state.current_question,
                answer,
                st.session_state.current_context,
                st.session_state.current_sources
            )

            st.session_state.feedback_text = (
                evaluation
            )

            save_question_log(
                student_id,
                st.session_state.current_subject,
                st.session_state.current_topic,
                st.session_state.current_question,
                answer,
                evaluation,
                score
            )

            st.session_state.last_score = score

            st.session_state.show_feedback = True

            st.rerun()
