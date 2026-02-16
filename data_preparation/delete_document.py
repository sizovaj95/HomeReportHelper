#!/usr/bin/env python3
import argparse
import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = "data_preparation/home_reports.db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete all SQLite records related to a specific document_id."
    )
    parser.add_argument("document_id", help="Document id to remove")
    parser.add_argument(
        "--sqlite-db",
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    return parser.parse_args()


def delete_document_records(db_path: str, document_id: str) -> dict[str, int]:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("BEGIN")
        try:
            cur.execute("DELETE FROM pipeline_runs WHERE document_id = ?", (document_id,))
            deleted_pipeline_runs = cur.rowcount

            cur.execute("DELETE FROM paragraphs WHERE document_id = ?", (document_id,))
            deleted_paragraphs = cur.rowcount

            cur.execute("DELETE FROM sections WHERE document_id = ?", (document_id,))
            deleted_sections = cur.rowcount

            cur.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            deleted_documents = cur.rowcount

            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return {
        "pipeline_runs": deleted_pipeline_runs,
        "paragraphs": deleted_paragraphs,
        "sections": deleted_sections,
        "documents": deleted_documents,
    }


def main() -> None:
    args = parse_args()
    deleted = delete_document_records(args.sqlite_db, args.document_id)

    print(f"document_id={args.document_id}")
    print(f"deleted pipeline_runs: {deleted['pipeline_runs']}")
    print(f"deleted paragraphs:    {deleted['paragraphs']}")
    print(f"deleted sections:      {deleted['sections']}")
    print(f"deleted documents:     {deleted['documents']}")

    if deleted["documents"] == 0:
        print("No document row found for this id.")


if __name__ == "__main__":
    main()
