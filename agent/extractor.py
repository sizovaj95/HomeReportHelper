from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from agent import config
from agent.graph import build_agent_graph
from agent.graph_state import GraphState
from agent.prompts import FIELD_SPECS, SYSTEM_PROMPT, make_user_prompt
from agent.retrieval import HybridRetriever
from agent.schema import (
    EvidenceParagraph,
    FieldResultModel,
    ListFieldResultModel,
    PropertyReportOutputModel,
)


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


class AgentExtractor:
    def __init__(self, retriever: HybridRetriever, model: str = config.EXTRACTION_MODEL):
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for extraction.") from exc

        if not OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY for extraction.")

        self.retriever = retriever
        self.model = model
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.graph = build_agent_graph(
            retriever=self.retriever,
            model=self.model,
            extract_field_from_candidates=self._extract_field_from_candidates,
            empty_result_factory=self._empty_result,
        )

    def extract_report(self, document_id: str, file_name: str) -> PropertyReportOutputModel:
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
            value = [str(v).strip() for v in raw_value if str(v).strip()]
            if not value and status == "found":
                status = "not_found"
            return ListFieldResultModel(
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
        if is_list:
            return ListFieldResultModel(status="not_found")
        return FieldResultModel(status="not_found")
