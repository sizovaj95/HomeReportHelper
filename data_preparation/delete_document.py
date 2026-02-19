#!/usr/bin/env python3
import argparse
import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = "data_preparation/home_reports.db"
DEFAULT_CHROMA_DIR = "data_preparation/chroma_db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete all records related to a specific document_id (SQLite + Chroma vectors)."
    )
    parser.add_argument("document_id", help="Document id to remove")
    parser.add_argument(
        "--sqlite-db",
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--chroma-dir",
        default=DEFAULT_CHROMA_DIR,
        help=f"Path to Chroma persistence directory (default: {DEFAULT_CHROMA_DIR})",
    )
    parser.add_argument(
        "--skip-chroma",
        action="store_true",
        help="Skip vector deletion from Chroma.",
    )
    return parser.parse_args()


def delete_document_records(db_path: str, document_id: str) -> dict[str, int]:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("BEGIN")
        try:
            cur.execute("DELETE FROM document_processing_status WHERE document_id = ?", (document_id,))
            deleted_processing_status = cur.rowcount

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
        "processing_status": deleted_processing_status,
        "pipeline_runs": deleted_pipeline_runs,
        "paragraphs": deleted_paragraphs,
        "sections": deleted_sections,
        "documents": deleted_documents,
    }


def delete_chroma_vectors(document_id: str, chroma_dir: str) -> int:
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "chromadb is not installed. Install it or run with --skip-chroma."
        ) from exc

    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(name="home_report_paragraphs_v1")

    fetched = collection.get(where={"document_id": document_id}, include=[])
    ids = fetched.get("ids", []) or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def main() -> None:
    args = parse_args()
    deleted_vectors = 0
    if not args.skip_chroma:
        deleted_vectors = delete_chroma_vectors(args.document_id, args.chroma_dir)

    deleted = delete_document_records(args.sqlite_db, args.document_id)

    print(f"document_id={args.document_id}")
    if args.skip_chroma:
        print("deleted chroma vectors: skipped")
    else:
        print(f"deleted chroma vectors: {deleted_vectors}")
    print(f"deleted processing status: {deleted['processing_status']}")
    print(f"deleted pipeline_runs: {deleted['pipeline_runs']}")
    print(f"deleted paragraphs:    {deleted['paragraphs']}")
    print(f"deleted sections:      {deleted['sections']}")
    print(f"deleted documents:     {deleted['documents']}")

    if deleted["documents"] == 0:
        print("No document row found for this id.")


if __name__ == "__main__":
    main()
