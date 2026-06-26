import os
import sqlite3
import streamlit as st
from datetime import datetime
from pathlib import Path

def get_project_root():
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "Home.py").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent

BASE_DIR = get_project_root()
DB_PATH  = str(BASE_DIR / "student_db" / "students.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    c    = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            class      TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(name, class)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS topic_scores (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            subject    TEXT NOT NULL,
            topic      TEXT NOT NULL,
            score      REAL NOT NULL,
            attempts   INTEGER DEFAULT 1,
            last_seen  TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            bot        TEXT NOT NULL,
            started_at TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS assessment_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            subject TEXT,
            topic TEXT,
            score REAL,
            total_questions INTEGER,
            difficulty TEXT,
            created_at TEXT
            )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS assessment_question_logs (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_id INTEGER,

            subject TEXT,

            topic TEXT,

            question TEXT,

            student_answer TEXT,

            evaluation TEXT,

            score REAL,

            created_at TEXT
        )
    """)
        

    conn.commit()
    conn.close()

def get_or_create_student(name: str, class_: str) -> int:
    """Return student_id, creating the student if needed."""
    conn = get_connection()
    c    = conn.cursor()

    c.execute(
        "SELECT id FROM students WHERE name=? AND class=?",
        (name, class_)
    )
    row = c.fetchone()

    if row:
        conn.close()
        return row[0]

    now = datetime.now().isoformat()
    c.execute(
        "INSERT INTO students (name, class, created_at) VALUES (?,?,?)",
        (name, class_, now)
    )
    conn.commit()
    student_id = c.lastrowid
    conn.close()
    return student_id

def get_student_class(student_id):

    conn = get_connection()
    c = conn.cursor()

    c.execute(
        """
        SELECT class
        FROM students
        WHERE id = ?
        """,
        (student_id,)
    )

    row = c.fetchone()

    conn.close()

    if row:
        return row[0]

    return None

def get_topics_for_class_subject(vectorstore,student_class,subject):

    topics = set()

    for doc in vectorstore.docstore._dict.values():

        if (
            doc.metadata.get("class") == student_class
            and
            doc.metadata.get("subject") == subject
        ):

            topic = doc.metadata.get("topic")

            if topic:
                topics.add(topic)

    return sorted(list(topics))

def update_topic_score(student_id: int, subject: str,
                       topic: str, score: float):
    """Insert or update a topic score."""
    conn = get_connection()
    c    = conn.cursor()
    now  = datetime.now().isoformat()

    c.execute("""
        SELECT id, score, attempts FROM topic_scores
        WHERE student_id=? AND subject=? AND topic=?
    """, (student_id, subject, topic))
    row = c.fetchone()

    if row:
        # Running average
        new_score    = (row[1] * row[2] + score) / (row[2] + 1)
        new_attempts = row[2] + 1
        c.execute("""
            UPDATE topic_scores
            SET score=?, attempts=?, last_seen=?
            WHERE id=?
        """, (new_score, new_attempts, now, row[0]))
    else:
        c.execute("""
            INSERT INTO topic_scores
            (student_id, subject, topic, score, attempts, last_seen)
            VALUES (?,?,?,?,1,?)
        """, (student_id, subject, topic, score, now))

    conn.commit()
    conn.close()

def save_assessment_attempt(
    student_id: int,
    subject: str,
    topic: str,
    score: float,
    total_questions: int,
    difficulty: str
):
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        INSERT INTO assessment_attempts
        (
            student_id,
            subject,
            topic,
            score,
            total_questions,
            difficulty,
            created_at
        )
        VALUES (?,?,?,?,?,?,?)
    """, (
        student_id,
        subject,
        topic,
        score,
        total_questions,
        difficulty,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()

def save_question_log(
    student_id,
    subject,
    topic,
    question,
    student_answer,
    evaluation,
    score
):

    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        INSERT INTO assessment_question_logs
        (
            student_id,
            subject,
            topic,
            question,
            student_answer,
            evaluation,
            score,
            created_at
        )
        VALUES (?,?,?,?,?,?,?,?)
    """, (
        student_id,
        subject,
        topic,
        question,
        student_answer,
        evaluation,
        score,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()

def get_assessment_history(student_id):

    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        SELECT
            subject,
            topic,
            score,
            difficulty,
            created_at
        FROM assessment_attempts
        WHERE student_id=?
        ORDER BY created_at DESC
    """, (student_id,))

    rows = c.fetchall()

    conn.close()

    return [
        {
            "subject": r[0],
            "topic": r[1],
            "score": r[2],
            "difficulty": r[3],
            "created_at": r[4]
        }
        for r in rows
    ]

def get_weak_topics(student_id: int, threshold: float = 0.6, subject: str = None) -> list:
    """Return topics where average score is below threshold."""
    conn = get_connection()
    c    = conn.cursor()

    if subject:

        c.execute("""
            SELECT
                subject,
                topic,
                score,
                attempts
            FROM topic_scores
            WHERE student_id=?
            AND subject=?
            AND score < ?
            ORDER BY score ASC
        """, (
            student_id,
            subject,
            threshold
        ))

    else:

        c.execute("""
            SELECT
                subject,
                topic,
                score,
                attempts
            FROM topic_scores
            WHERE student_id=?
            AND score < ?
            ORDER BY score ASC
        """, (
            student_id,
            threshold
        ))
    rows = c.fetchall()
    conn.close()

    return [
        {"subject": r[0], "topic": r[1],
         "score": round(r[2], 2), "attempts": r[3]}
        for r in rows
    ]

def get_all_scores(student_id: int) -> list:
    """Return all topic scores for a student."""
    conn = get_connection()
    c    = conn.cursor()

    c.execute("""
        SELECT subject, topic, score, attempts, last_seen
        FROM topic_scores
        WHERE student_id=?
        ORDER BY subject, topic
    """, (student_id,))
    rows = c.fetchall()
    conn.close()

    return [
        {"subject": r[0], "topic": r[1], "score": round(r[2], 2),
         "attempts": r[3], "last_seen": r[4]}
        for r in rows
    ]

def get_topic_performance(student_id):

    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        SELECT
            subject,
            topic,
            score,
            attempts
        FROM topic_scores
        WHERE student_id=?
        ORDER BY score ASC
    """, (student_id,))

    rows = c.fetchall()

    conn.close()

    return rows
    
def get_subjects_for_class(vectorstore, student_class):
    """
    Return all unique subjects available for a given class
    from the FAISS metadata.
    """
    subjects = set()

    for doc in vectorstore.docstore._dict.values():

        if doc.metadata.get("class") != student_class:
            continue

        subject = doc.metadata.get("subject")

        if subject:
            subjects.add(subject)

    return sorted(subjects)
