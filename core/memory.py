"""Persistent memory — SQLite-backed conversation & knowledge storage"""

import os
import sqlite3
from typing import Optional

class ConversationMemory:
    """Persistent conversation memory for agents"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "..", "agent_memory.db")
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id, id);

            CREATE TABLE IF NOT EXISTS knowledge (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS agent_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                note TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        conn.close()

    def add_message(self, session_id: str, role: str, content: str):
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()
        conn.close()

    def get_history(self, session_id: str, limit: int = 30) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        conn.close()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    def remember(self, key: str, value: str):
        """Store a key-value knowledge"""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO knowledge (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()
        conn.close()

    def recall(self, key: str) -> Optional[str]:
        """Retrieve knowledge by key"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM knowledge WHERE key = ?", (key,)
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def add_note(self, agent_name: str, note: str):
        """Agent can save a note for future reference"""
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO agent_notes (agent_name, note) VALUES (?, ?)",
            (agent_name, note),
        )
        conn.commit()
        conn.close()

    def get_notes(self, agent_name: str, limit: int = 10) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT note FROM agent_notes WHERE agent_name = ? ORDER BY id DESC LIMIT ?",
            (agent_name, limit),
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]

    def clear_session(self, session_id: str):
        conn = self._get_conn()
        conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
