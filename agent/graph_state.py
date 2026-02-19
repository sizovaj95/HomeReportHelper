from __future__ import annotations

from typing import TypedDict

from agent.models import CandidateChunk
from agent.schema import FieldResultModel, ListFieldResultModel, PropertyReportOutputModel


class GraphState(TypedDict, total=False):
    document_id: str
    file_name: str
    model: str
    field_keys: list[str]
    current_field_index: int
    current_field_key: str | None
    current_candidates: list[CandidateChunk]
    current_result: FieldResultModel | ListFieldResultModel | None
    field_results: dict[str, FieldResultModel | ListFieldResultModel]
    errors: list[str]
    started_at: str
    generated_at: str | None
    output: PropertyReportOutputModel | None
