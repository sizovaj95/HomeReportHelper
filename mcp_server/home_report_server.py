from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent import config
from agent.retrieval import HybridRetriever
from agent.storage import AgentStorage
from data_preparation.pipeline_service import prepare_document_if_needed
from data_preparation.storage_sqlite import SQLiteStore

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "MCP server requires the 'mcp' package. Add it to your environment (pip install mcp)."
    ) from exc


logger = logging.getLogger(__name__)

SERVER_NAME = "home-report-domain"
REPORTS_DIR = config.REPORTS_DIR

mcp = FastMCP(SERVER_NAME)


def _sqlite_store() -> SQLiteStore:
    store = SQLiteStore(config.SQLITE_DB_PATH)
    store.ensure_schema()
    return store


def _agent_storage() -> AgentStorage:
    return AgentStorage(config.SQLITE_DB_PATH)


def _retriever() -> HybridRetriever:
    return HybridRetriever(storage=_agent_storage(), chroma_dir=config.CHROMA_DIR)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_report_path(file_name: str) -> Path:
    candidate = (REPORTS_DIR / file_name).resolve()
    reports_root = REPORTS_DIR.resolve()
    if reports_root not in candidate.parents and candidate != reports_root:
        raise ValueError("file_name resolves outside reports directory")
    if not candidate.exists():
        raise FileNotFoundError(f"Report not found: {file_name}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Not a file: {file_name}")
    return candidate


def _serialize_section(section) -> dict[str, Any]:
    return {
        "section_id": section.section_id,
        "document_id": section.document_id,
        "section_order": section.section_order,
        "title": section.title,
        "summary": section.summary,
        "pages": list(section.pages),
    }


def _serialize_paragraph(paragraph) -> dict[str, Any]:
    return {
        "paragraph_id": paragraph.paragraph_id,
        "document_id": paragraph.document_id,
        "section_id": paragraph.section_id,
        "order_in_section": paragraph.order_in_section,
        "text": paragraph.text,
        "pages": list(paragraph.pages),
        "is_heading_like": bool(paragraph.is_heading_like),
    }


def _serialize_candidate(candidate) -> dict[str, Any]:
    return asdict(candidate)


@mcp.tool()
def list_reports() -> list[dict[str, Any]]:
    """List available PDF reports in the local reports folder."""
    if not REPORTS_DIR.exists():
        return []

    files = []
    for path in sorted(REPORTS_DIR.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".pdf":
            continue
        files.append(
            {
                "file_name": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return files


@mcp.tool()
def get_document_status(file_name: str) -> dict[str, Any]:
    """Get current preparation status for a report file by name."""
    pdf_path = _resolve_report_path(file_name)
    sqlite_store = _sqlite_store()
    file_sha256 = _sha256_file(pdf_path)
    status = sqlite_store.get_document_processing_status(file_sha256)

    return {
        "file_name": pdf_path.name,
        "file_path": str(pdf_path),
        "file_sha256": file_sha256,
        "status": status,
    }


@mcp.tool()
async def prepare_report(
    file_name: str,
    run_summaries: bool = True,
    run_embeddings: bool = True,
) -> dict[str, Any]:
    """Run preparation if needed; reuses cached canonical data if already prepared."""
    pdf_path = _resolve_report_path(file_name)
    # Run blocking prep in a worker thread to avoid event-loop conflicts.
    info = await asyncio.to_thread(
        prepare_document_if_needed,
        pdf_path=pdf_path,
        sqlite_db=config.SQLITE_DB_PATH,
        chroma_dir=config.CHROMA_DIR,
        run_summaries=run_summaries,
        run_embeddings=run_embeddings,
    )
    return {
        "document_id": info.document_id,
        "file_name": info.file_name,
        "file_sha256": info.file_sha256,
        "was_prepared_now": info.was_prepared_now,
    }


@mcp.tool()
def get_section_overview(document_id: str) -> list[dict[str, Any]]:
    """Return section ids, titles, summaries, and pages for a prepared document."""
    retriever = _retriever()
    return retriever.get_section_overview(document_id)


@mcp.tool()
def retrieve_candidates_hybrid(
    document_id: str,
    query_hints: list[str],
    top_k_vector: int | None = None,
    top_k_keyword: int | None = None,
    final_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Run hybrid retrieval (keyword + section summary + vector) for query hints."""
    retriever = _retriever()
    candidates = retriever.retrieve_candidates(
        document_id=document_id,
        query_hints=query_hints,
        top_k_vector=top_k_vector or config.RETRIEVAL_TOP_K_VECTOR,
        top_k_keyword=top_k_keyword or config.RETRIEVAL_TOP_K_KEYWORD,
        final_limit=final_limit or config.FINAL_CANDIDATE_LIMIT,
    )
    return [_serialize_candidate(c) for c in candidates]


@mcp.tool()
def retrieve_candidates_from_sections(
    document_id: str,
    section_ids: list[str],
    query_hints: list[str],
    final_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Retrieve paragraph candidates only from selected sections."""
    retriever = _retriever()
    candidates = retriever.retrieve_candidates_from_sections(
        document_id=document_id,
        section_ids=section_ids,
        query_hints=query_hints,
        final_limit=final_limit or config.FINAL_CANDIDATE_LIMIT,
    )
    return [_serialize_candidate(c) for c in candidates]


@mcp.tool()
def get_paragraphs_by_ids(document_id: str, paragraph_ids: list[str]) -> list[dict[str, Any]]:
    """Fetch exact paragraphs by IDs for evidence display."""
    if not paragraph_ids:
        return []
    storage = _agent_storage()
    para_map = {p.paragraph_id: p for p in storage.get_paragraphs(document_id)}
    out: list[dict[str, Any]] = []
    for pid in paragraph_ids:
        paragraph = para_map.get(pid)
        if paragraph is None:
            continue
        out.append(_serialize_paragraph(paragraph))
    return out


@mcp.tool()
def get_sections(document_id: str) -> list[dict[str, Any]]:
    """Return full stored sections for a document."""
    storage = _agent_storage()
    return [_serialize_section(s) for s in storage.get_sections(document_id)]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger.info("Starting MCP server '%s' (reports_dir=%s)", SERVER_NAME, REPORTS_DIR)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
