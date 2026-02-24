from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Mapping

from dotenv import load_dotenv

from agent import config
from agent.graph import build_agent_graph
from agent.graph_state import GraphState
from agent.prompts import (
    FIELD_SPECS,
    SECTION_ROUTER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    make_section_router_prompt,
    make_user_prompt,
)
from agent.retrieval import HybridRetriever
from agent.schema import (
    EvidenceParagraph,
    FieldResultModel,
    PropertyReportOutputModel,
)


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
logger = logging.getLogger(__name__)
GREEN = "\033[92m"
RESET = "\033[0m"


class AgentExtractor:
    def __init__(
        self,
        retriever: HybridRetriever,
        models: Mapping[str, str] | None = None,
    ):
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for extraction.") from exc

        if not OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY for extraction.")

        default_models = {
            "EXTRACTOR_MODEL": config.EXTRACTION_MODEL,
            "SECTION_MODEL": config.SECTION_MODEL,
        }
        if models:
            default_models.update({k: v for k, v in models.items() if v})

        self.models = default_models
        self.retriever = retriever
        self.model = self.models["EXTRACTOR_MODEL"]
        self.section_model = self.models["SECTION_MODEL"]
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self._cached_section_overview: list[dict] | None = None
        self.graph = build_agent_graph(
            retriever=self.retriever,
            model=self.model,
            retrieve_candidates_for_field=self._retrieve_candidates_for_field,
            extract_field_from_candidates=self._extract_field_from_candidates,
            empty_result_factory=self._empty_result,
        )

    def extract_report(self, document_id: str, file_name: str) -> PropertyReportOutputModel:
        # Reset per-document cache at the beginning of each report run.
        self._cached_section_overview = None

        initial_state: GraphState = {
            "document_id": document_id,
            "file_name": file_name,
            "model": self.model,
            "field_keys": list(FIELD_SPECS.keys()),
            "current_field_index": 0,
            "current_field_key": None,
            "current_candidates": [],
            "field_results": {},
            "errors": [],
            "started_at": datetime.now(tz=timezone.utc).isoformat(),
            "generated_at": None,
            "output": None,
        }
        # This graph iterates several nodes per field, so default recursion_limit=25 is too low.
        per_field_steps = 4  # retrieve -> extract -> store -> set_next
        base_steps = 6       # init/start/finalize overhead
        recursion_limit = max(50, (len(FIELD_SPECS) * per_field_steps) + base_steps)
        final_state = self.graph.invoke(initial_state, config={"recursion_limit": recursion_limit})
        output = final_state.get("output")
        if output is None:
            # Defensive fallback: should not happen if graph finalize node ran.
            output = PropertyReportOutputModel(
                file_name=file_name,
                generated_at=datetime.now(tz=timezone.utc).isoformat(),
                model_used=self.model,
                document_id=document_id,
            )
        return output

    def _retrieve_candidates_for_field(
        self,
        document_id: str,
        field_key: str,
        field_label: str,
        query_hints: list[str],
    ):
        logger.info("%sExtracting field: %s%s", GREEN, field_key, RESET)
        limits = self._compute_retrieval_limits(query_hints)
        logger.info(
            "%sField %s: query_count=%s final_limit=%s top_k_vector=%s top_k_keyword=%s%s",
            GREEN,
            field_key,
            limits["query_count"],
            limits["final_limit"],
            limits["top_k_vector"],
            limits["top_k_keyword"],
            RESET,
        )
        if self._cached_section_overview is None:
            self._cached_section_overview = self.retriever.get_section_overview(document_id)

        section_overview = self._cached_section_overview
        selected_section_ids = self._select_relevant_sections(
            field_key=field_key,
            field_label=field_label,
            section_overview=section_overview,
        )

        section_routed = self.retriever.retrieve_candidates_from_sections(
            document_id=document_id,
            section_ids=selected_section_ids,
            query_hints=query_hints,
            final_limit=limits["final_limit"],
        )
        logger.info(
            "%sField %s: section_routed candidates=%s (selected_sections=%s)%s",
            GREEN,
            field_key,
            len(section_routed),
            len(selected_section_ids),
            RESET,
        )
        hybrid = self.retriever.retrieve_candidates(
            document_id=document_id,
            query_hints=query_hints,
            top_k_vector=limits["top_k_vector"],
            top_k_keyword=limits["top_k_keyword"],
            final_limit=limits["final_limit"],
        )
        logger.info("%sField %s: hybrid candidates=%s%s", GREEN, field_key, len(hybrid), RESET)

        merged = self._merge_candidates(section_routed, hybrid)
        trimmed = merged[:limits["final_limit"]]
        logger.info(
            "%sField %s: merged unique candidates=%s -> trimmed=%s%s",
            GREEN,
            field_key,
            len(merged),
            len(trimmed),
            RESET,
        )
        return trimmed

    def _select_relevant_sections(
        self,
        field_key: str,
        field_label: str,
        section_overview: list[dict],
    ) -> list[str]:
        if not section_overview:
            return []

        overview_text = "\n".join(
            f"section_id={s['section_id']} | title={s['title'] or 'N/A'} | summary={s['summary'] or 'N/A'}"
            for s in section_overview
        )
        allowed_ids = {s["section_id"] for s in section_overview}

        try:
            response = self.client.chat.completions.create(
                model=self.section_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SECTION_ROUTER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": make_section_router_prompt(
                            field_key,
                            field_label,
                            overview_text,
                        ),
                    },
                ],
            )
            content = response.choices[0].message.content or "{}"
            parsed = self._safe_json_loads(content)
            section_ids = parsed.get("section_ids", [])
            if not isinstance(section_ids, list):
                return []
            filtered = [sid for sid in section_ids if isinstance(sid, str) and sid in allowed_ids]
            seen = set()
            deduped = []
            for sid in filtered:
                if sid in seen:
                    continue
                seen.add(sid)
                deduped.append(sid)
            return deduped[:5]
        except Exception:
            return []

    def _merge_candidates(self, primary, fallback):
        merged = []
        seen = set()
        for candidate in primary + fallback:
            if candidate.paragraph_id in seen:
                continue
            seen.add(candidate.paragraph_id)
            merged.append(candidate)
        return merged

    def _compute_retrieval_limits(self, query_hints: list[str]) -> dict[str, int]:
        if hasattr(self.retriever, "prepare_unique_queries"):
            unique_queries = self.retriever.prepare_unique_queries(query_hints)
        else:
            unique_queries = self._normalize_unique_queries(query_hints)

        query_count = max(1, len(unique_queries))
        final_limit = min(
            config.FINAL_CANDIDATE_LIMIT_MAX,
            config.FINAL_CANDIDATE_LIMIT_BASE + (config.FINAL_CANDIDATE_LIMIT_PER_QUERY * query_count),
        )
        top_k_vector = max(config.RETRIEVAL_TOP_K_VECTOR, final_limit)
        top_k_keyword = max(config.RETRIEVAL_TOP_K_KEYWORD, final_limit + config.RETRIEVAL_KEYWORD_BUFFER)

        return {
            "query_count": query_count,
            "final_limit": final_limit,
            "top_k_vector": top_k_vector,
            "top_k_keyword": top_k_keyword,
        }

    def _normalize_unique_queries(self, query_hints: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for hint in query_hints:
            value = " ".join((hint or "").strip().lower().split())
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    def _extract_field_from_candidates(
        self,
        field_key: str,
        field_label: str,
        is_list: bool,
        candidates,
    ):
        if not candidates:
            return self._empty_result(is_list=is_list)

        evidence_text = "\n\n".join(
            f"paragraph_id={c.paragraph_id} page={c.page if c.page is not None else 'NA'} source={c.source}\n{c.text}"
            for c in candidates
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": make_user_prompt(field_key, field_label, is_list, evidence_text),
                },
            ],
        )

        content = response.choices[0].message.content or "{}"
        parsed = self._safe_json_loads(content)

        evidence_ids = parsed.get("evidence_paragraph_ids", [])
        evidence_rows = [c for c in candidates if c.paragraph_id in evidence_ids]

        found_pages = self._to_int_list(parsed.get("found_pages", []))
        candidate_pages = self._to_int_list(parsed.get("candidate_pages", []))

        if not found_pages:
            found_pages = sorted({c.page for c in evidence_rows if c.page is not None})
        if not candidate_pages and parsed.get("status") == "not_found":
            candidate_pages = sorted({c.page for c in candidates if c.page is not None})[:5]

        evidence_models = [
            EvidenceParagraph(
                paragraph_id=c.paragraph_id,
                page=c.page,
                text=c.text,
                relevance_note=f"Retrieved via {c.source}",
            )
            for c in evidence_rows
        ]

        status = parsed.get("status")
        if status not in {"found", "not_found", "ambiguous"}:
            status = "not_found"

        confidence = parsed.get("confidence")
        if not isinstance(confidence, (float, int)):
            confidence = None

        if is_list:
            raw_value = parsed.get("value", [])
            if not isinstance(raw_value, list):
                raw_value = []
            value_items = [str(v).strip() for v in raw_value if str(v).strip()]
            if not value_items and status == "found":
                status = "not_found"
            value = self._to_bullet_string(value_items) if value_items else None
            return FieldResultModel(
                value=value,
                status=status,
                found_pages=found_pages,
                candidate_pages=candidate_pages,
                evidence_paragraphs=evidence_models,
                confidence=float(confidence) if confidence is not None else None,
            )

        value = parsed.get("value")
        if value is not None:
            value = str(value).strip() or None
        if value is None and status == "found":
            status = "not_found"

        return FieldResultModel(
            value=value,
            status=status,
            found_pages=found_pages,
            candidate_pages=candidate_pages,
            evidence_paragraphs=evidence_models,
            confidence=float(confidence) if confidence is not None else None,
        )

    def _safe_json_loads(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _to_int_list(self, values) -> list[int]:
        if not isinstance(values, list):
            return []
        out: list[int] = []
        for v in values:
            try:
                out.append(int(v))
            except (TypeError, ValueError):
                continue
        return sorted(set(out))

    def _empty_result(self, is_list: bool):
        return FieldResultModel(status="not_found")

    def _to_bullet_string(self, items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items if item)
