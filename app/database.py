# app/database.py
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path("quiz_results.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            student TEXT,
            answer TEXT,
            score REAL,
            feedback TEXT
        );
        """
    )
    conn.commit()
    conn.close()

def store_results(graded):
    conn = sqlite3.connect(DB_PATH)
    now = datetime.utcnow().isoformat()
    for g in graded:
        conn.execute(
            "INSERT INTO results (timestamp, question, student, answer, score, feedback) VALUES (?, ?, ?, ?, ?, ?)",
            (now, g["question"], g["student"], g["answer"], g["score"], g["feedback"]),
        )
    conn.commit()
    conn.close()
