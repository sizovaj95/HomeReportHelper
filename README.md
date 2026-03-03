# Home Report Helper

AI-assisted extraction pipeline for Scottish Home Reports (PDF):

1. Build canonical document representation (sections + paragraphs + summaries + embeddings).
2. Retrieve grounded evidence from local stores (SQLite + ChromaDB).
3. Extract structured fields with LLM and render JSON + HTML outputs with page/paragraph evidence.

## What This Project Does

- Ingests Home Report PDFs from `data_preparation/reports/`.
- Uses Azure Document Intelligence (`prebuilt-layout`) for layout-aware OCR.
- Normalizes documents into canonical records:
  - `documents`
  - `sections`
  - `paragraphs`
- Generates section descriptions and paragraph embeddings (OpenAI).
- Runs a LangGraph-based extraction agent that:
  - retrieves relevant evidence,
  - extracts fields conservatively (no guessing),
  - outputs JSON and HTML reports.
- Supports MCP-based tool access via a custom domain server.

## Repository Structure

- `data_preparation/`
  - Layout processing, canonicalization, summaries, embeddings, storage.
  - Main service entrypoint: `data_preparation/pipeline_service.py`.
- `agent/`
  - Retrieval + extraction + HTML rendering.
  - Main runtime entrypoint: `agent/run.py`.
- `mcp_server/`
  - Custom MCP server exposing domain tools.
  - Entrypoint: `mcp_server/home_report_server.py`.

## Data Flow

1. **Preparation** (`prepare_document_if_needed`)
- Compute file hash.
- If already prepared, reuse existing record.
- Otherwise:
  - OCR/layout via Azure,
  - split/clean sections and paragraphs,
  - store canonical records in SQLite,
  - generate summaries and embeddings,
  - store vectors in ChromaDB.

2. **Extraction Agent** (`agent/run.py`)
- Select report interactively.
- Ensure preparation exists.
- For each field:
  - retrieve candidate paragraphs,
  - route via section summaries,
  - extract value using LLM with strict evidence grounding.
- Render:
  - JSON (`agent/output/*.json`)
  - HTML (`agent/output/*.html`)

## Requirements

- Python 3.12+
- Azure Document Intelligence credentials
- OpenAI API key

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Configuration

Create/update `.env` with the required keys:

```env
# Azure Document Intelligence
AZURE_LANGUAGE_SERVICE_ENDPOINT=...
AZURE_LANGUAGE_SERVICE_API_KEY=...

# OpenAI
OPENAI_API_KEY=...

# Optional model overrides
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

Agent model selection is configured in `agent/config.py`:

- `EXTRACTION_MODEL` (default: `gpt-4.1-mini`)
- `SECTION_MODEL` (default: `gpt-4o-mini`)

## Run Modes

### 1) Main Agent Run (recommended)

```bash
python3 -m agent.run
```

What happens:
- prompts you to pick a report from `data_preparation/reports/`
- prepares document if needed
- runs extraction
- writes outputs to `agent/output/`

### 2) Preparation Only

```bash
python3 data_preparation/prepare_document.py
```

Optional flags:

```bash
python3 data_preparation/prepare_document.py \
  --pdf-path data_preparation/reports/LochburnGardensGlasgow.pdf \
  --skip-summaries \
  --skip-embeddings
```

## MCP Integration

This project includes a custom domain MCP server.

Start server manually:

```bash
python3 -m mcp_server.home_report_server
```

Run agent using real MCP client (stdio):

```bash
AGENT_USE_MCP_TOOLS=1 python3 -m agent.run
```

Notes:
- In MCP mode, `agent/run.py` calls tools through MCP protocol.
- MCP session is persistent for the whole run.
- If MCP fails to initialize, agent falls back to direct local mode.

## Storage

### SQLite

Default DB: `data_preparation/home_reports.db`

Main tables:
- `documents`
- `sections`
- `paragraphs`
- `pipeline_runs`
- `document_processing_status`

### ChromaDB

Default directory: `data_preparation/chroma_db`

Collection:
- `home_report_paragraphs_v1`

## Outputs

Generated in `agent/output/`:

- `<document_name>__<timestamp>.json`
- `<document_name>__<timestamp>.html`

HTML includes:
- grouped field sections,
- status/value per field,
- found/candidate pages,
- collapsible evidence paragraphs.

## Utility Scripts

Delete one document record footprint:

```bash
python3 data_preparation/delete_document.py <DOCUMENT_ID>
```
