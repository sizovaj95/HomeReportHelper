# Home Report MCP Server

Minimal domain MCP server for this project. It exposes report/preparation/retrieval tools over MCP stdio.

## Run

```bash
python3 -m mcp_server.home_report_server
```

## Use MCP-backed current agent flow

Your existing `agent/run.py` can now use a **real MCP client over stdio** for prep + retrieval:

```bash
AGENT_USE_MCP_TOOLS=1 python3 -m agent.run
```

In this mode, the agent starts `python3 -m mcp_server.home_report_server` as an MCP server subprocess and calls tools via MCP protocol.
The MCP stdio session is kept open for the whole run (not per tool call).

If MCP client or server cannot load, `agent/run.py` falls back to direct local mode automatically.

## Exposed tools (v1)

- `list_reports`
- `get_document_status`
- `prepare_report`
- `get_section_overview`
- `get_sections`
- `retrieve_candidates_hybrid`
- `retrieve_candidates_from_sections`
- `get_paragraphs_by_ids`

## Notes

- `prepare_report` wraps existing `data_preparation.pipeline_service.prepare_document_if_needed(...)`.
- Retrieval tools use the same SQLite + Chroma stores as the current agent.
- This is a minimal rollout. Your existing `agent/run.py` flow remains unchanged.
