import hashlib
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from data_preparation import layout_objects as lo
from data_preparation.embeddings import EmbeddingClient
from data_preparation.schema import DocumentRecordModel, ParagraphRecordModel, SectionRecordModel
from data_preparation.split_layout import LayoutProcessor
from data_preparation.storage_chroma import ChromaStore
from data_preparation.storage_sqlite import SQLiteStore
from data_preparation.summarize_sections import SectionSummarizer


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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


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
    logger.info("Stage 1/9: Starting document preparation from pickled layout")
    layout_pkl_path = Path(LAYOUT_PKL_PATH)
    if not layout_pkl_path.exists():
        raise FileNotFoundError(f"Layout pkl not found: {layout_pkl_path}")
    logger.info("Using layout file: %s", layout_pkl_path)

    logger.info("Stage 2/9: Initializing SQLite store and schema")
    sqlite_store = SQLiteStore(SQLITE_DB_PATH)
    sqlite_store.ensure_schema()

    logger.info("Stage 3/9: Checking if document was already processed")
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

    logger.info("Stage 4/9: Loading layout object from pickle")
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
    logger.info("Prepared document record document_id=%s source=%s", document.document_id, source_name)

    logger.info("Stage 5/9: Processing layout into sections and paragraphs")
    processor = LayoutProcessor(layout=layout, document_id=document.document_id)
    sections, paragraphs = processor.process()
    logger.info("Layout processing done: sections=%s paragraphs=%s", len(sections), len(paragraphs))

    section_text_map = {
        section.section_id: processor.build_section_text_for_summary(section)
        for section in sections
    }

    logger.info("Stage 6/9: Validating canonical records")
    validate_records(document, sections, paragraphs)
    logger.info("Validation complete")

    logger.info("Stage 7/9: Persisting document, sections, and paragraphs to SQLite")
    sqlite_store.upsert_document(document)
    sqlite_store.upsert_sections(sections)
    sqlite_store.upsert_paragraphs(paragraphs)
    logger.info("SQLite persistence complete")

    if not SKIP_SUMMARIES:
        logger.info("Stage 8/9: Generating section descriptions")
        summarizer = SectionSummarizer()
        summarizer.summarize_sections(sections, section_text_map)
        sqlite_store.upsert_sections(sections)
        logger.info("Section descriptions generated and persisted")
    else:
        logger.info("Stage 8/9: Summaries skipped by config")

    if not SKIP_EMBEDDINGS:
        logger.info("Stage 9/9: Generating embeddings and storing vectors in Chroma")
        embedding_client = EmbeddingClient()
        chroma_store = ChromaStore(CHROMA_DIR)
        vectors_by_id, token_count_by_id, embedding_model = embedding_client.embed_paragraphs(paragraphs)
        logger.info("Embeddings generated for %s paragraphs", len(vectors_by_id))

        section_order_by_id = {section.section_id: section.section_order for section in sections}
        vector_id_map = chroma_store.upsert_paragraph_vectors(
            paragraphs=paragraphs,
            vectors_by_paragraph_id=vectors_by_id,
            section_order_by_id=section_order_by_id,
        )
        logger.info("Chroma upsert complete: vectors=%s", len(vector_id_map))

        for paragraph in paragraphs:
            paragraph.embedding_model = embedding_model
            paragraph.embedding_vector_id = vector_id_map.get(paragraph.paragraph_id)
            paragraph.token_count = token_count_by_id.get(paragraph.paragraph_id)

        sqlite_store.upsert_paragraphs(paragraphs)
        logger.info("Updated paragraph embedding metadata in SQLite")
    else:
        logger.info("Stage 9/9: Embeddings skipped by config")

    summary = {
        "document_id": document.document_id,
        "source": source_name,
        "layout_pkl": str(layout_pkl_path),
        "sections": len(sections),
        "paragraphs": len(paragraphs),
        "summaries_enabled": not SKIP_SUMMARIES,
        "embeddings_enabled": not SKIP_EMBEDDINGS,
    }
    logger.info("Process complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
