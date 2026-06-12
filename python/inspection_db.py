import json
import os
import sqlite3
from datetime import datetime


DB_PATH = "/userdata/inspection.db"
RESULTS_DIR = "/userdata/results"


def init_db(db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspector TEXT NOT NULL,
            location TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            status TEXT DEFAULT '进行中'
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            result_path TEXT NOT NULL,
            confidence REAL,
            crack_count INTEGER DEFAULT 0,
            crack_area INTEGER DEFAULT 0,
            max_width REAL DEFAULT 0,
            total_length REAL DEFAULT 0,
            severity TEXT,
            suggestion TEXT,
            result_json TEXT,
            captured_at TEXT NOT NULL,
            uploaded INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    conn.commit()
    conn.close()


def create_session(inspector, location, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO sessions (inspector, location, start_time, status) VALUES (?, ?, ?, ?)",
        (inspector, location, start_time, "进行中")
    )
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id, start_time


def end_session(session_id, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "UPDATE sessions SET end_time = ?, status = ? WHERE id = ?",
        (end_time, "已完成", session_id)
    )
    conn.commit()
    conn.close()
    return end_time


def save_capture(session_id, image_path, result_path, detection_data, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    captured_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO captures (
            session_id, image_path, result_path, confidence, crack_count,
            crack_area, max_width, total_length, severity, suggestion,
            result_json, captured_at, uploaded
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
    """, (
        session_id,
        image_path,
        result_path,
        detection_data.get("confidence", 0),
        detection_data.get("crack_count", 0),
        detection_data.get("crack_area", 0),
        detection_data.get("max_width", 0),
        detection_data.get("total_length", 0),
        detection_data.get("severity", "无"),
        detection_data.get("suggestion", ""),
        json.dumps(detection_data.get("result_json", {}), ensure_ascii=False),
        captured_at,
    ))

    capture_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return capture_id


def get_session(session_id, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_session_captures(session_id, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM captures WHERE session_id = ? ORDER BY captured_at ASC",
        (session_id,)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_unuploaded_captures(session_id=None, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if session_id is not None:
        cursor.execute(
            "SELECT * FROM captures WHERE uploaded = 0 AND session_id = ? ORDER BY captured_at ASC",
            (session_id,)
        )
    else:
        cursor.execute(
            "SELECT * FROM captures WHERE uploaded = 0 ORDER BY captured_at ASC"
        )

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def mark_capture_uploaded(capture_id, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE captures SET uploaded = 1 WHERE id = ?", (capture_id,))
    conn.commit()
    conn.close()
