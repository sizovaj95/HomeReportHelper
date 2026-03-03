from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    # Allow running as a script: `python agent/run.py` from repo root or from `agent/`.
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from agent import config
from agent.extractor import AgentExtractor
from agent.html_renderer import render_html
from agent.mcp_stdio_client import MCPStdioClient
from agent.retrieval import HybridRetriever
from agent.retrieval_mcp import MCPHybridRetriever
from agent.storage import AgentStorage
from data_preparation.pipeline_service import prepare_document_if_needed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def choose_pdf_from_reports_dir(reports_dir: Path) -> Path:
    if not reports_dir.exists() or not reports_dir.is_dir():
        raise FileNotFoundError(f"Reports folder not found: {reports_dir}")

    pdf_files = sorted(reports_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in reports folder: {reports_dir}")

    print("Available reports:")
    for idx, pdf in enumerate(pdf_files, start=1):
        print(f"{idx}. {pdf.name}")

    while True:
        choice = input("Pick a report by number: ").strip()
        if not choice.isdigit():
            print("Please enter a number.")
            continue

        number = int(choice)
        if number < 1 or number > len(pdf_files):
            print(f"Please choose a number between 1 and {len(pdf_files)}.")
            continue
        return pdf_files[number - 1]


def sanitize_filename_base(file_name: str) -> str:
    base = Path(file_name).stem
    base = re.sub(r"[^A-Za-z0-9]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    return base or "document"


def build_output_paths(file_name: str, now: datetime) -> tuple[Path, Path]:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_base = sanitize_filename_base(file_name)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    json_path = config.OUTPUT_DIR / f"{safe_base}__{timestamp}.json"
    html_path = config.OUTPUT_DIR / f"{safe_base}__{timestamp}.html"
    return json_path, html_path


def main() -> None:
    logger.info("Stage 1/6: Starting agent run")
    logger.info("Looking for reports in: %s", config.REPORTS_DIR)
    pdf_path = choose_pdf_from_reports_dir(config.REPORTS_DIR)
    logger.info("Selected document: %s", pdf_path.name)
    use_mcp_tools = os.getenv("AGENT_USE_MCP_TOOLS", "0").strip().lower() in {"1", "true", "yes"}
    mcp_client = None
    if use_mcp_tools:
        try:
            mcp_client = MCPStdioClient()
            logger.info("MCP mode enabled (AGENT_USE_MCP_TOOLS=1): using real MCP stdio client")
        except Exception as exc:
            logger.warning("Failed to load MCP tools, falling back to direct mode: %s", exc)
            use_mcp_tools = False

    try:
        logger.info("Stage 2/6: Checking cache and running data preparation if needed")
        if use_mcp_tools and mcp_client is not None:
            mcp_retriever = MCPHybridRetriever(mcp_client)
            try:
                prep_info = mcp_retriever.prepare_report(file_name=pdf_path.name)
                logger.info(
                    "Preparation status (MCP): was_prepared_now=%s document_id=%s",
                    prep_info.was_prepared_now,
                    prep_info.document_id,
                )
                retriever = mcp_retriever
            except Exception as exc:
                logger.warning("MCP preparation call failed, falling back to direct mode: %s", exc)
                use_mcp_tools = False

        if not use_mcp_tools:
            prep_info = prepare_document_if_needed(
                pdf_path=pdf_path,
                sqlite_db=config.SQLITE_DB_PATH,
                chroma_dir=config.CHROMA_DIR,
            )
            logger.info(
                "Preparation status: was_prepared_now=%s document_id=%s",
                prep_info.was_prepared_now,
                prep_info.document_id,
            )

            logger.info("Stage 3/6: Initializing storage and retrieval components")
            storage = AgentStorage(config.SQLITE_DB_PATH)
            retriever = HybridRetriever(storage=storage, chroma_dir=config.CHROMA_DIR)

        if use_mcp_tools:
            logger.info("Stage 3/6: Initializing MCP-backed retrieval components")
        extractor = AgentExtractor(
            retriever=retriever,
            models={
                "EXTRACTOR_MODEL": config.EXTRACTION_MODEL,
                "SECTION_MODEL": config.SECTION_MODEL,
            },
        )

        logger.info("Stage 4/6: Extracting required fields using grounded evidence")
        report = extractor.extract_report(
            document_id=prep_info.document_id,
            file_name=prep_info.file_name,
        )
        logger.info("Extraction complete for file: %s", prep_info.file_name)

        logger.info("Stage 5/6: Rendering HTML and preparing output paths")
        html = render_html(report)
        now = datetime.now()
        json_path, html_path = build_output_paths(prep_info.file_name, now)

        logger.info("Stage 6/6: Writing JSON and HTML artifacts")
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        html_path.write_text(html, encoding="utf-8")
        logger.info("Wrote JSON output: %s", json_path)
        logger.info("Wrote HTML output: %s", html_path)

        print(
            json.dumps(
                {
                    "document_id": prep_info.document_id,
                    "file_name": prep_info.file_name,
                    "was_prepared_now": prep_info.was_prepared_now,
                    "json_output": str(json_path),
                    "html_output": str(html_path),
                },
                indent=2,
            )
        )
        logger.info("Agent run complete")
    finally:
        if mcp_client is not None:
            mcp_client.close()


if __name__ == "__main__":
    main()
