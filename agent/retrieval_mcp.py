from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from agent.models import CandidateChunk

logger = logging.getLogger(__name__)


@dataclass
class MCPPreparedInfo:
    document_id: str
    file_name: str
    file_sha256: str
    was_prepared_now: bool


class MCPHybridRetriever:
    """Retriever adapter backed by MCP domain tools."""

    def __init__(self, mcp_client):
        self._client = mcp_client

    def _tool(self, name: str, arguments: dict) -> dict | list:
        payload = self._client.call_tool(name, arguments)
        normalized = self._normalize_payload(payload)
        if isinstance(normalized, (dict, list)):
            return normalized
        raise RuntimeError(f"MCP tool '{name}' returned unsupported payload type: {type(payload).__name__}")

    def get_section_overview(self, document_id: str) -> list[dict]:
        payload = self._tool("get_section_overview", {"document_id": document_id})
        rows = self._as_list_payload(payload)
        if not isinstance(rows, list):
            logger.error("Unexpected payload for get_section_overview: type=%s value=%r", type(payload).__name__, payload)
            raise RuntimeError("MCP get_section_overview did not return a list")
        return rows

    def retrieve_candidates(
        self,
        document_id: str,
        query_hints: list[str],
        top_k_vector: int,
        top_k_keyword: int,
        final_limit: int,
    ) -> list[CandidateChunk]:
        payload = self._tool(
            "retrieve_candidates_hybrid",
            {
                "document_id": document_id,
                "query_hints": query_hints,
                "top_k_vector": top_k_vector,
                "top_k_keyword": top_k_keyword,
                "final_limit": final_limit,
            },
        )
        rows = self._as_list_payload(payload)
        if not isinstance(rows, list):
            logger.error("Unexpected payload for retrieve_candidates_hybrid: type=%s value=%r", type(payload).__name__, payload)
            raise RuntimeError("MCP retrieve_candidates_hybrid did not return a list")
        return [self._to_candidate(row) for row in rows]

    def retrieve_candidates_from_sections(
        self,
        document_id: str,
        section_ids: list[str],
        query_hints: list[str],
        final_limit: int,
    ) -> list[CandidateChunk]:
        payload = self._tool(
            "retrieve_candidates_from_sections",
            {
                "document_id": document_id,
                "section_ids": section_ids,
                "query_hints": query_hints,
                "final_limit": final_limit,
            },
        )
        rows = self._as_list_payload(payload)
        if not isinstance(rows, list):
            logger.error(
                "Unexpected payload for retrieve_candidates_from_sections: type=%s value=%r",
                type(payload).__name__,
                payload,
            )
            raise RuntimeError("MCP retrieve_candidates_from_sections did not return a list")
        return [self._to_candidate(row) for row in rows]

    def prepare_unique_queries(self, query_hints: list[str]) -> list[str]:
        # Keep consistent normalization contract with local HybridRetriever.
        seen: set[str] = set()
        out: list[str] = []
        for hint in query_hints:
            normalized = " ".join((hint or "").strip().lower().split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
        return out

    def prepare_report(
        self,
        file_name: str,
        run_summaries: bool = True,
        run_embeddings: bool = True,
    ) -> MCPPreparedInfo:
        payload = self._tool(
            "prepare_report",
            {
                "file_name": file_name,
                "run_summaries": run_summaries,
                "run_embeddings": run_embeddings,
            },
        )
        payload = self._as_dict_payload(payload)
        if not isinstance(payload, dict):
            raise RuntimeError("MCP prepare_report did not return an object")
        return MCPPreparedInfo(
            document_id=str(payload["document_id"]),
            file_name=str(payload["file_name"]),
            file_sha256=str(payload["file_sha256"]),
            was_prepared_now=bool(payload["was_prepared_now"]),
        )

    @staticmethod
    def _to_candidate(row: dict) -> CandidateChunk:
        return CandidateChunk(
            paragraph_id=str(row["paragraph_id"]),
            section_id=str(row["section_id"]),
            page=int(row["page"]) if row.get("page") is not None else None,
            text=str(row["text"]),
            source=str(row["source"]),
            score=float(row["score"]),
        )

    def _normalize_payload(self, payload: Any) -> Any:
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return text
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return payload
        return payload

    def _as_list_payload(self, payload: Any) -> list[Any] | None:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("result", "data", "items", "rows", "value", "payload"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        return None

    def _as_dict_payload(self, payload: Any) -> dict[str, Any] | None:
        if isinstance(payload, dict):
            for key in ("result", "data", "item", "value", "payload"):
                value = payload.get(key)
                if isinstance(value, dict):
                    return value
            return payload
        return None
