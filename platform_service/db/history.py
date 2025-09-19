# platform_service/db/history.py

"""
history.py
대화 기록을 SQLite DB에 저장하고 조회하는 모듈
"""
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
import logging

from platform_service import constants


logger = logging.getLogger(__name__)
DB_PATH = constants.KB_DATA_DIR / "history.db"

# =============================================================
# DB 초기화
# =============================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        with conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
    finally:
        conn.close()

# =============================================================
# 메시지 저장
# =============================================================
def save_message(session_id: str, role: str, message: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        with conn:
            conn.execute(
                "INSERT INTO chat_history (session_id, role, message) VALUES (?, ?, ?)",
                (session_id, role, message)
            )
    except Exception as e:
        logger.error(f"[History] 메시지 저장 실패: {e}")
    finally:
        conn.close()

# =============================================================
# 메시지 조회
# =============================================================
def load_messages(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, message, created_at FROM chat_history WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit)
        )
        rows = cursor.fetchall()
        return [{"role": r, "message": m, "created_at": t} for r, m, t in rows]
    except Exception as e:
        logger.error(f"[History] 메시지 로드 실패: {e}")
        return []
    finally:
        conn.close()
