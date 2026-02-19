from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StoredParagraph:
    paragraph_id: str
    document_id: str
    section_id: str
    order_in_section: int
    text: str
    pages: list[int]
    is_heading_like: bool


@dataclass
class StoredSection:
    section_id: str
    document_id: str
    section_order: int
    title: str | None
    summary: str
    pages: list[int]


class AgentStorage:
    def __init__(self, sqlite_db_path: str):
        self.sqlite_db_path = sqlite_db_path
        Path(sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def list_documents(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT d.document_id, d.file_name, d.file_sha256, d.created_at,
                       COALESCE(s.cnt, 0) AS section_count,
                       COALESCE(p.cnt, 0) AS paragraph_count
                FROM documents d
                LEFT JOIN (
                    SELECT document_id, COUNT(*) AS cnt FROM sections GROUP BY document_id
                ) s ON s.document_id = d.document_id
                LEFT JOIN (
                    SELECT document_id, COUNT(*) AS cnt FROM paragraphs GROUP BY document_id
                ) p ON p.document_id = d.document_id
                ORDER BY d.created_at DESC
                """
            ).fetchall()

    def get_document_by_sha(self, file_sha256: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM documents WHERE file_sha256 = ? ORDER BY created_at DESC LIMIT 1",
                (file_sha256,),
            ).fetchone()

    def get_document(self, document_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()

    def get_sections(self, document_id: str) -> list[StoredSection]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sections WHERE document_id = ? ORDER BY section_order",
                (document_id,),
            ).fetchall()

        return [
            StoredSection(
                section_id=row["section_id"],
                document_id=row["document_id"],
                section_order=row["section_order"],
                title=row["title"],
                summary=row["summary"],
                pages=json.loads(row["pages_json"]),
            )
            for row in rows
        ]

    def get_paragraphs(self, document_id: str) -> list[StoredParagraph]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM paragraphs WHERE document_id = ? ORDER BY section_id, order_in_section",
                (document_id,),
            ).fetchall()

        return [
            StoredParagraph(
                paragraph_id=row["paragraph_id"],
                document_id=row["document_id"],
                section_id=row["section_id"],
                order_in_section=row["order_in_section"],
                text=row["text"],
                pages=json.loads(row["pages_json"]),
                is_heading_like=bool(row["is_heading_like"]),
            )
            for row in rows
        ]

    def keyword_search_paragraphs(self, document_id: str, terms: list[str], limit: int) -> list[StoredParagraph]:
        if not terms:
            return []

        query_parts = ["text LIKE ?" for _ in terms]
        params = [f"%{term}%" for term in terms]
        where_clause = " OR ".join(query_parts)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM paragraphs
                WHERE document_id = ?
                  AND ({where_clause})
                ORDER BY section_id, order_in_section
                LIMIT ?
                """,
                (document_id, *params, limit),
            ).fetchall()

        return [
            StoredParagraph(
                paragraph_id=row["paragraph_id"],
                document_id=row["document_id"],
                section_id=row["section_id"],
                order_in_section=row["order_in_section"],
                text=row["text"],
                pages=json.loads(row["pages_json"]),
                is_heading_like=bool(row["is_heading_like"]),
            )
            for row in rows
        ]

    def keyword_search_sections(self, document_id: str, terms: list[str], limit: int) -> list[StoredSection]:
        if not terms:
            return []

        query_parts = ["summary LIKE ?" for _ in terms]
        params = [f"%{term}%" for term in terms]
        where_clause = " OR ".join(query_parts)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM sections
                WHERE document_id = ?
                  AND ({where_clause})
                ORDER BY section_order
                LIMIT ?
                """,
                (document_id, *params, limit),
            ).fetchall()

        return [
            StoredSection(
                section_id=row["section_id"],
                document_id=row["document_id"],
                section_order=row["section_order"],
                title=row["title"],
                summary=row["summary"],
                pages=json.loads(row["pages_json"]),
            )
            for row in rows
        ]
