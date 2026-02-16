import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import layout_objects as lo
from embeddings import EmbeddingClient
from schema import DocumentRecordModel, ParagraphRecordModel, SectionRecordModel
from split_layout import LayoutProcessor
from storage_chroma import ChromaStore
from storage_sqlite import SQLiteStore
from summarize_sections import SectionSummarizer


DATA_PREPARATION_DIR = Path(__file__).resolve().parent
DEFAULT_LAYOUT_PKL = DATA_PREPARATION_DIR / "example_layout.pkl"
# Script config (edit here when needed).
LAYOUT_PKL_PATH = DEFAULT_LAYOUT_PKL
SOURCE_NAME: str | None = None
DOCUMENT_ID: str | None = None
SKIP_SUMMARIES = False
SKIP_EMBEDDINGS = False
SQLITE_DB_PATH = "data_preparation/home_reports.db"
CHROMA_DIR = "data_preparation/chroma_db"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def build_document_record(
    source_name: str,
    layout,
    file_sha256: str,
    document_id: str | None = None,
) -> lo.DocumentRecord:
    return lo.DocumentRecord(
        document_id=document_id or str(uuid4()),
        schema_version=lo.SCHEMA_VERSION,
        file_name=source_name,
        file_sha256=file_sha256,
        page_count=len(layout.pages or []),
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


def validate_records(
    document: lo.DocumentRecord,
    sections: list[lo.SectionRecord],
    paragraphs: list[lo.ParagraphRecord],
) -> None:
    DocumentRecordModel(**document.__dict__)
    for section in sections:
        SectionRecordModel(**section.__dict__)
    for paragraph in paragraphs:
        ParagraphRecordModel(**paragraph.__dict__)


def main() -> None:
    layout_pkl_path = Path(LAYOUT_PKL_PATH)
    if not layout_pkl_path.exists():
        raise FileNotFoundError(f"Layout pkl not found: {layout_pkl_path}")

    sqlite_store = SQLiteStore(SQLITE_DB_PATH)
    sqlite_store.ensure_schema()

    layout_sha256 = sha256_file(layout_pkl_path)
    processing_status = sqlite_store.get_document_processing_status(layout_sha256)
    if processing_status and processing_status["canonical_exists"]:
        print(
            json.dumps(
                {
                    "status": "already_processed",
                    "document_id": processing_status["document_id"],
                    "file_name": processing_status["file_name"],
                    "created_at": processing_status["created_at"],
                    "sections": processing_status["section_count"],
                    "paragraphs": processing_status["paragraph_count"],
                },
                indent=2,
            )
        )
        return

    with layout_pkl_path.open("rb") as fp:
        layout = pickle.load(fp)

    existing_document_id = None
    if processing_status and not processing_status["canonical_exists"]:
        existing_document_id = processing_status["document_id"]

    source_name = SOURCE_NAME or layout_pkl_path.name
    document = build_document_record(
        source_name=source_name,
        layout=layout,
        file_sha256=layout_sha256,
        document_id=DOCUMENT_ID or existing_document_id,
    )

    processor = LayoutProcessor(layout=layout, document_id=document.document_id)
    sections, paragraphs = processor.process()

    section_text_map = {
        section.section_id: processor.build_section_text_for_summary(section)
        for section in sections
    }

    validate_records(document, sections, paragraphs)

    sqlite_store.upsert_document(document)
    sqlite_store.upsert_sections(sections)
    sqlite_store.upsert_paragraphs(paragraphs)

    if not SKIP_SUMMARIES:
        summarizer = SectionSummarizer()
        summarizer.summarize_sections(sections, section_text_map)
        sqlite_store.upsert_sections(sections)

    if not SKIP_EMBEDDINGS:
        embedding_client = EmbeddingClient()
        chroma_store = ChromaStore(CHROMA_DIR)
        vectors_by_id, token_count_by_id, embedding_model = embedding_client.embed_paragraphs(paragraphs)

        section_order_by_id = {section.section_id: section.section_order for section in sections}
        vector_id_map = chroma_store.upsert_paragraph_vectors(
            paragraphs=paragraphs,
            vectors_by_paragraph_id=vectors_by_id,
            section_order_by_id=section_order_by_id,
        )

        for paragraph in paragraphs:
            paragraph.embedding_model = embedding_model
            paragraph.embedding_vector_id = vector_id_map.get(paragraph.paragraph_id)
            paragraph.token_count = token_count_by_id.get(paragraph.paragraph_id)

        sqlite_store.upsert_paragraphs(paragraphs)

    summary = {
        "document_id": document.document_id,
        "source": source_name,
        "layout_pkl": str(layout_pkl_path),
        "sections": len(sections),
        "paragraphs": len(paragraphs),
        "summaries_enabled": not SKIP_SUMMARIES,
        "embeddings_enabled": not SKIP_EMBEDDINGS,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
