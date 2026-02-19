from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

from data_preparation import layout_objects as lo
from data_preparation.embeddings import EmbeddingClient, MIN_EMBEDDING_WORDS
from data_preparation.schema import DocumentRecordModel, ParagraphRecordModel, SectionRecordModel
from data_preparation.split_layout import LayoutProcessor
from data_preparation.storage_chroma import ChromaStore
from data_preparation.storage_sqlite import SQLiteStore
from data_preparation.summarize_sections import SectionSummarizer


load_dotenv()

AZURE_LANG_ENDPOINT = os.getenv("AZURE_LANGUAGE_SERVICE_ENDPOINT", "")
AZURE_LANG_API_KEY = os.getenv("AZURE_LANGUAGE_SERVICE_API_KEY", "")


@dataclass
class PreparedDocumentInfo:
    document_id: str
    file_name: str
    file_sha256: str
    was_prepared_now: bool


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
    pdf_path: Path,
    layout,
    document_id: str | None = None,
    file_sha256: str | None = None,
) -> lo.DocumentRecord:
    return lo.DocumentRecord(
        document_id=document_id or str(uuid4()),
        schema_version=lo.SCHEMA_VERSION,
        file_name=pdf_path.name,
        file_sha256=file_sha256 or sha256_file(pdf_path),
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


def build_section_text_map(
    sections: list[lo.SectionRecord],
    paragraphs: list[lo.ParagraphRecord],
) -> dict[str, str]:
    para_map = {p.paragraph_id: p for p in paragraphs}
    section_text_map: dict[str, str] = {}

    for section in sections:
        chunks: list[str] = []
        if section.inherited_headings:
            chunks.append("Inherited headings:\n" + "\n".join(section.inherited_headings))

        body_text: list[str] = []
        for paragraph_id in section.paragraph_ids:
            paragraph = para_map.get(paragraph_id)
            if paragraph is None:
                continue
            if paragraph.is_heading_like and paragraph.kind == lo.PARAGRAPH_KIND_TEXT:
                continue
            if paragraph.text:
                body_text.append(paragraph.text)
        if body_text:
            chunks.append("\n\n".join(body_text))

        section_text_map[section.section_id] = "\n\n".join(chunks).strip()

    return section_text_map


def count_embeddable_paragraphs(paragraphs: list[lo.ParagraphRecord]) -> int:
    return sum(
        1
        for p in paragraphs
        if (not p.is_heading_like) and len((p.text or "").strip().split()) >= MIN_EMBEDDING_WORDS
    )


def prepare_document_if_needed(
    pdf_path: Path,
    sqlite_db: str = "data_preparation/home_reports.db",
    chroma_dir: str = "data_preparation/chroma_db",
    run_summaries: bool = True,
    run_embeddings: bool = True,
) -> PreparedDocumentInfo:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    sqlite_store = SQLiteStore(sqlite_db)
    sqlite_store.ensure_schema()

    pdf_sha256 = sha256_file(pdf_path)
    processing_status = sqlite_store.get_document_processing_status(pdf_sha256)
    existing_document_id = processing_status["document_id"] if processing_status else None
    canonical_ready = processing_status["canonical_ready"] if processing_status else False
    summaries_ready = processing_status["summaries_ready"] if processing_status else False
    embeddings_ready = processing_status["embeddings_ready"] if processing_status else False
    file_name = processing_status["file_name"] if processing_status else pdf_path.name

    did_any_work = False

    if canonical_ready:
        sections = sqlite_store.load_sections(existing_document_id)
        paragraphs = sqlite_store.load_paragraphs(existing_document_id)
        section_text_map = build_section_text_map(sections, paragraphs)

        if run_summaries and not summaries_ready:
            did_any_work = True
            try:
                summarizer = SectionSummarizer()
                summarizer.summarize_sections(sections, section_text_map)
                sqlite_store.upsert_sections(sections)
                sqlite_store.set_document_processing_status(
                    existing_document_id,
                    summaries_ready=True,
                    last_error=None,
                )
                summaries_ready = True
            except Exception as exc:
                sqlite_store.set_document_processing_status(
                    existing_document_id,
                    summaries_ready=False,
                    last_error=f"summaries_failed: {type(exc).__name__}: {exc}",
                )
                raise

        if run_embeddings and not embeddings_ready:
            did_any_work = True
            try:
                embedding_client = EmbeddingClient()
                chroma_store = ChromaStore(chroma_dir)
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

                embeddable_count = count_embeddable_paragraphs(paragraphs)
                completed_count = len(vectors_by_id)
                fully_ready = completed_count >= embeddable_count
                sqlite_store.set_document_processing_status(
                    existing_document_id,
                    embeddings_ready=fully_ready,
                    last_error=None if fully_ready else (
                        f"embeddings_partial: generated {completed_count}/{embeddable_count}"
                    ),
                )
                embeddings_ready = fully_ready
            except Exception as exc:
                sqlite_store.set_document_processing_status(
                    existing_document_id,
                    embeddings_ready=False,
                    last_error=f"embeddings_failed: {type(exc).__name__}: {exc}",
                )
                raise

        if (not run_summaries or summaries_ready) and (not run_embeddings or embeddings_ready):
            return PreparedDocumentInfo(
                document_id=existing_document_id,
                file_name=file_name,
                file_sha256=pdf_sha256,
                was_prepared_now=did_any_work,
            )

    if not AZURE_LANG_ENDPOINT or not AZURE_LANG_API_KEY:
        raise RuntimeError("Missing Azure Language Service credentials.")

    client = DocumentIntelligenceClient(
        endpoint=AZURE_LANG_ENDPOINT,
        credential=AzureKeyCredential(AZURE_LANG_API_KEY),
    )

    doc_bytes = pdf_path.read_bytes()
    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        body=AnalyzeDocumentRequest(bytes_source=doc_bytes),
        output=["figures"],
    )
    layout = poller.result()

    document = build_document_record(
        pdf_path,
        layout,
        existing_document_id,
        file_sha256=pdf_sha256,
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
    sqlite_store.set_document_processing_status(
        document.document_id,
        canonical_ready=True,
        summaries_ready=False,
        embeddings_ready=False,
        last_error=None,
    )

    if run_summaries:
        try:
            summarizer = SectionSummarizer()
            summarizer.summarize_sections(sections, section_text_map)
            sqlite_store.upsert_sections(sections)
            sqlite_store.set_document_processing_status(
                document.document_id,
                summaries_ready=True,
                last_error=None,
            )
        except Exception as exc:
            sqlite_store.set_document_processing_status(
                document.document_id,
                summaries_ready=False,
                last_error=f"summaries_failed: {type(exc).__name__}: {exc}",
            )
            raise

    if run_embeddings:
        try:
            embedding_client = EmbeddingClient()
            chroma_store = ChromaStore(chroma_dir)
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

            embeddable_count = count_embeddable_paragraphs(paragraphs)
            completed_count = len(vectors_by_id)
            fully_ready = completed_count >= embeddable_count
            sqlite_store.set_document_processing_status(
                document.document_id,
                embeddings_ready=fully_ready,
                last_error=None if fully_ready else (
                    f"embeddings_partial: generated {completed_count}/{embeddable_count}"
                ),
            )
        except Exception as exc:
            sqlite_store.set_document_processing_status(
                document.document_id,
                embeddings_ready=False,
                last_error=f"embeddings_failed: {type(exc).__name__}: {exc}",
            )
            raise

    return PreparedDocumentInfo(
        document_id=document.document_id,
        file_name=document.file_name,
        file_sha256=document.file_sha256,
        was_prepared_now=True,
    )
