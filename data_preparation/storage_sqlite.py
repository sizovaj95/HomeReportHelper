import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import layout_objects as lo


class SQLiteStore:
    def __init__(self, db_path: str | Path = "data_preparation/home_reports.db"):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_sha256 TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sections (
                    section_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    section_order INTEGER NOT NULL,
                    title TEXT,
                    summary TEXT NOT NULL,
                    boundary_source TEXT NOT NULL,
                    pages_json TEXT NOT NULL,
                    paragraph_ids_json TEXT NOT NULL,
                    merged_from_section_ids_json TEXT NOT NULL,
                    is_heading_only_original INTEGER NOT NULL,
                    inherited_headings_json TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );

                CREATE TABLE IF NOT EXISTS paragraphs (
                    paragraph_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    section_id TEXT NOT NULL,
                    order_in_section INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    pages_json TEXT NOT NULL,
                    layout_refs_json TEXT NOT NULL,
                    role TEXT,
                    is_heading_like INTEGER NOT NULL,
                    merged_from_ids_json TEXT NOT NULL,
                    embedding_model TEXT,
                    embedding_vector_id TEXT,
                    token_count INTEGER,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id),
                    FOREIGN KEY(section_id) REFERENCES sections(section_id)
                );

                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    step TEXT NOT NULL,
                    model TEXT,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );

                CREATE INDEX IF NOT EXISTS idx_documents_file_sha256 ON documents(file_sha256);
                CREATE INDEX IF NOT EXISTS idx_sections_document_id ON sections(document_id);
                CREATE INDEX IF NOT EXISTS idx_paragraphs_document_id ON paragraphs(document_id);
                """
            )

    def upsert_document(self, doc: lo.DocumentRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (document_id, schema_version, file_name, file_sha256, page_count, created_at)
                VALUES (:document_id, :schema_version, :file_name, :file_sha256, :page_count, :created_at)
                ON CONFLICT(document_id) DO UPDATE SET
                    schema_version=excluded.schema_version,
                    file_name=excluded.file_name,
                    file_sha256=excluded.file_sha256,
                    page_count=excluded.page_count,
                    created_at=excluded.created_at
                """,
                asdict(doc),
            )

    def upsert_sections(self, sections: list[lo.SectionRecord]) -> None:
        if not sections:
            return

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO sections (
                    section_id, document_id, section_order, title, summary, boundary_source,
                    pages_json, paragraph_ids_json, merged_from_section_ids_json,
                    is_heading_only_original, inherited_headings_json
                ) VALUES (
                    :section_id, :document_id, :section_order, :title, :summary, :boundary_source,
                    :pages_json, :paragraph_ids_json, :merged_from_section_ids_json,
                    :is_heading_only_original, :inherited_headings_json
                )
                ON CONFLICT(section_id) DO UPDATE SET
                    document_id=excluded.document_id,
                    section_order=excluded.section_order,
                    title=excluded.title,
                    summary=excluded.summary,
                    boundary_source=excluded.boundary_source,
                    pages_json=excluded.pages_json,
                    paragraph_ids_json=excluded.paragraph_ids_json,
                    merged_from_section_ids_json=excluded.merged_from_section_ids_json,
                    is_heading_only_original=excluded.is_heading_only_original,
                    inherited_headings_json=excluded.inherited_headings_json
                """,
                [
                    {
                        "section_id": s.section_id,
                        "document_id": s.document_id,
                        "section_order": s.section_order,
                        "title": s.title,
                        "summary": s.summary,
                        "boundary_source": s.boundary_source,
                        "pages_json": json.dumps(s.pages),
                        "paragraph_ids_json": json.dumps(s.paragraph_ids),
                        "merged_from_section_ids_json": json.dumps(s.merged_from_section_ids),
                        "is_heading_only_original": int(s.is_heading_only_original),
                        "inherited_headings_json": json.dumps(s.inherited_headings),
                    }
                    for s in sections
                ],
            )

    def upsert_paragraphs(self, paragraphs: list[lo.ParagraphRecord]) -> None:
        if not paragraphs:
            return

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO paragraphs (
                    paragraph_id, document_id, section_id, order_in_section, kind, text,
                    pages_json, layout_refs_json, role, is_heading_like,
                    merged_from_ids_json, embedding_model, embedding_vector_id, token_count
                ) VALUES (
                    :paragraph_id, :document_id, :section_id, :order_in_section, :kind, :text,
                    :pages_json, :layout_refs_json, :role, :is_heading_like,
                    :merged_from_ids_json, :embedding_model, :embedding_vector_id, :token_count
                )
                ON CONFLICT(paragraph_id) DO UPDATE SET
                    document_id=excluded.document_id,
                    section_id=excluded.section_id,
                    order_in_section=excluded.order_in_section,
                    kind=excluded.kind,
                    text=excluded.text,
                    pages_json=excluded.pages_json,
                    layout_refs_json=excluded.layout_refs_json,
                    role=excluded.role,
                    is_heading_like=excluded.is_heading_like,
                    merged_from_ids_json=excluded.merged_from_ids_json,
                    embedding_model=excluded.embedding_model,
                    embedding_vector_id=excluded.embedding_vector_id,
                    token_count=excluded.token_count
                """,
                [
                    {
                        "paragraph_id": p.paragraph_id,
                        "document_id": p.document_id,
                        "section_id": p.section_id,
                        "order_in_section": p.order_in_section,
                        "kind": p.kind,
                        "text": p.text,
                        "pages_json": json.dumps(p.pages),
                        "layout_refs_json": json.dumps(p.layout_refs),
                        "role": p.role,
                        "is_heading_like": int(p.is_heading_like),
                        "merged_from_ids_json": json.dumps(p.merged_from_ids),
                        "embedding_model": p.embedding_model,
                        "embedding_vector_id": p.embedding_vector_id,
                        "token_count": p.token_count,
                    }
                    for p in paragraphs
                ],
            )

    def upsert_pipeline_run(
        self,
        run_id: str,
        document_id: str,
        step: str,
        model: str | None,
        status: str,
        started_at: str,
        finished_at: str | None,
        error_message: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pipeline_runs (
                    run_id, document_id, step, model, status, error_message, started_at, finished_at
                ) VALUES (
                    :run_id, :document_id, :step, :model, :status, :error_message, :started_at, :finished_at
                )
                ON CONFLICT(run_id) DO UPDATE SET
                    document_id=excluded.document_id,
                    step=excluded.step,
                    model=excluded.model,
                    status=excluded.status,
                    error_message=excluded.error_message,
                    started_at=excluded.started_at,
                    finished_at=excluded.finished_at
                """,
                {
                    "run_id": run_id,
                    "document_id": document_id,
                    "step": step,
                    "model": model,
                    "status": status,
                    "error_message": error_message,
                    "started_at": started_at,
                    "finished_at": finished_at,
                },
            )

    def list_paragraphs_for_document(self, document_id: str) -> list[sqlite3.Row]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM paragraphs WHERE document_id = ? ORDER BY section_id, order_in_section",
                (document_id,),
            ).fetchall()
        return rows

    def find_document_by_sha256(self, file_sha256: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM documents
                WHERE file_sha256 = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (file_sha256,),
            ).fetchone()
        return row

    def canonical_representation_exists(self, document_id: str) -> bool:
        with self._connect() as conn:
            section_count = conn.execute(
                "SELECT COUNT(1) AS c FROM sections WHERE document_id = ?",
                (document_id,),
            ).fetchone()["c"]
            paragraph_count = conn.execute(
                "SELECT COUNT(1) AS c FROM paragraphs WHERE document_id = ?",
                (document_id,),
            ).fetchone()["c"]
        return section_count > 0 and paragraph_count > 0

    def get_document_processing_status(self, file_sha256: str) -> dict | None:
        row = self.find_document_by_sha256(file_sha256)
        if row is None:
            return None

        document_id = row["document_id"]
        with self._connect() as conn:
            section_count = conn.execute(
                "SELECT COUNT(1) AS c FROM sections WHERE document_id = ?",
                (document_id,),
            ).fetchone()["c"]
            paragraph_count = conn.execute(
                "SELECT COUNT(1) AS c FROM paragraphs WHERE document_id = ?",
                (document_id,),
            ).fetchone()["c"]

        return {
            "document_id": document_id,
            "file_name": row["file_name"],
            "file_sha256": row["file_sha256"],
            "created_at": row["created_at"],
            "section_count": section_count,
            "paragraph_count": paragraph_count,
            "canonical_exists": section_count > 0 and paragraph_count > 0,
        }
