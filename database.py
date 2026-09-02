import sqlite3
import json
import contextlib
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np

DEFAULT_DB_PATH = "attendance.db"

@contextlib.contextmanager
def get_db_cursor(db_path: str = DEFAULT_DB_PATH):
    """Context manager that yields a cursor and guarantees connection closure."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    finally:
        conn.close()

def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initializes SQLite database schema for workers and attendance logs.
    Creates necessary tables and indexes if they do not already exist.
    """
    with get_db_cursor(db_path) as cursor:
        # Workers table: stores worker ID, name, serialized 128-d face encoding, and enrollment timestamp
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                face_encoding TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Attendance table: stores check-in records linked to a worker ID
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                worker_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                FOREIGN KEY (worker_id) REFERENCES workers (id) ON DELETE CASCADE
            )
        """)
        
        # Index on worker_id and timestamp for rapid cooldown lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_attendance_worker_time 
            ON attendance(worker_id, timestamp DESC)
        """)

def insert_worker(name: str, face_encoding: np.ndarray, db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Inserts a newly enrolled worker into the database.
    
    Args:
        name: Worker full name.
        face_encoding: 128-dimensional numpy vector representing facial features.
        db_path: Path to the SQLite database file.
        
    Returns:
        The auto-generated worker ID.
    """
    # ponytail: serialize 128-d numpy array as JSON list for lightweight, portable text storage
    encoding_json = json.dumps(face_encoding.tolist())
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with get_db_cursor(db_path) as cursor:
        cursor.execute(
            "INSERT INTO workers (name, face_encoding, created_at) VALUES (?, ?, ?)",
            (name.strip(), encoding_json, created_at)
        )
        return cursor.lastrowid

def get_all_workers(db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """
    Fetches all registered workers with their deserialized face encodings.
    
    Returns:
        List of dicts: [{'id': int, 'name': str, 'encoding': np.ndarray, 'created_at': str}]
    """
    with get_db_cursor(db_path) as cursor:
        cursor.execute("SELECT id, name, face_encoding, created_at FROM workers ORDER BY name ASC")
        rows = cursor.fetchall()
        
    workers = []
    for row in rows:
        worker_id, name, encoding_json, created_at = row
        encoding_array = np.array(json.loads(encoding_json), dtype=np.float64)
        workers.append({
            "id": worker_id,
            "name": name,
            "encoding": encoding_array,
            "created_at": created_at
        })
    return workers

def delete_worker(worker_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Deletes a worker and their associated attendance records."""
    with get_db_cursor(db_path) as cursor:
        cursor.execute("DELETE FROM workers WHERE id = ?", (worker_id,))
        return cursor.rowcount > 0

def log_attendance(worker_id: int, cooldown_minutes: int = 5, db_path: str = DEFAULT_DB_PATH) -> Tuple[bool, str]:
    """
    Logs attendance for a worker, enforcing a cooldown window to prevent rapid duplicate logs.
    
    Args:
        worker_id: The ID of the worker recognized.
        cooldown_minutes: Minimum minutes that must elapse before logging again (default: 5).
        db_path: Path to database.
        
    Returns:
        Tuple of (is_logged: bool, status_message: str)
    """
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    today_str = now.strftime("%Y-%m-%d")
    
    with get_db_cursor(db_path) as cursor:
        # Check the latest check-in for this worker
        cursor.execute(
            "SELECT timestamp FROM attendance WHERE worker_id = ? ORDER BY id DESC LIMIT 1",
            (worker_id,)
        )
        row = cursor.fetchone()
        
        if row:
            last_timestamp_str = row[0]
            try:
                last_time = datetime.strptime(last_timestamp_str, "%Y-%m-%d %H:%M:%S")
                elapsed_seconds = (now - last_time).total_seconds()
                cooldown_seconds = cooldown_minutes * 60
                
                if elapsed_seconds < cooldown_seconds:
                    remaining_mins = int((cooldown_seconds - elapsed_seconds) // 60) + 1
                    return False, f"Cooldown active. Already logged {int(elapsed_seconds // 60)}m ago. Cooldown resets in ~{remaining_mins}m."
            except ValueError:
                pass  # Fall through to log if timestamp parsing fails

        # Insert new attendance entry
        cursor.execute(
            "INSERT INTO attendance (worker_id, timestamp, date) VALUES (?, ?, ?)",
            (worker_id, now_str, today_str)
        )
        return True, f"Attendance successfully marked at {now.strftime('%H:%M:%S')}."

def get_attendance_logs(date_str: Optional[str] = None, db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """
    Retrieves attendance records joined with worker names.
    
    Args:
        date_str: Optional date filter in 'YYYY-MM-DD' format. If None, fetches all logs.
        db_path: Path to database.
        
    Returns:
        List of dictionaries with attendance log fields.
    """
    with get_db_cursor(db_path) as cursor:
        if date_str:
            cursor.execute("""
                SELECT a.id, a.worker_id, w.name, a.timestamp, a.date
                FROM attendance a
                JOIN workers w ON a.worker_id = w.id
                WHERE a.date = ?
                ORDER BY a.id DESC
            """, (date_str,))
        else:
            cursor.execute("""
                SELECT a.id, a.worker_id, w.name, a.timestamp, a.date
                FROM attendance a
                JOIN workers w ON a.worker_id = w.id
                ORDER BY a.id DESC
            """)
        rows = cursor.fetchall()
        
    return [
        {
            "Record ID": r[0],
            "Worker ID": r[1],
            "Worker Name": r[2],
            "Timestamp": r[3],
            "Date": r[4]
        }
        for r in rows
    ]

def get_worker_count(db_path: str = DEFAULT_DB_PATH) -> int:
    """Returns the total number of enrolled workers."""
    with get_db_cursor(db_path) as cursor:
        cursor.execute("SELECT COUNT(*) FROM workers")
        return cursor.fetchone()[0]
