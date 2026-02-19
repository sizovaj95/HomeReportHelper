from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from langgraph.graph import END, START, StateGraph

from agent.graph_state import GraphState
from agent.prompts import FIELD_SPECS


def build_agent_graph(
    retriever,
    model: str,
    extract_field_from_candidates: Callable,
    empty_result_factory: Callable,
):
    def init_report(state: GraphState) -> GraphState:
        return {
            "model": model,
            "field_keys": list(FIELD_SPECS.keys()),
            "current_field_index": 0,
            "current_field_key": None,
            "current_candidates": [],
            "current_result": None,
            "field_results": {},
            "errors": [],
            "generated_at": None,
            "output": None,
        }

    def set_next_field(state: GraphState) -> GraphState:
        idx = state["current_field_index"]
        keys = state["field_keys"]
        if idx >= len(keys):
            return {"current_field_key": None}
        return {"current_field_key": keys[idx], "current_candidates": [], "current_result": None}

    def retrieve_field_candidates(state: GraphState) -> GraphState:
        field_key = state.get("current_field_key")
        if not field_key:
            return {"current_candidates": []}

        spec = FIELD_SPECS[field_key]
        try:
            candidates = retriever.retrieve_candidates(state["document_id"], spec["queries"])
            return {"current_candidates": candidates}
        except Exception as exc:
            errors = list(state.get("errors", []))
            errors.append(f"retrieval_failed:{field_key}:{type(exc).__name__}:{exc}")
            return {"current_candidates": [], "errors": errors}

    def extract_field_value(state: GraphState) -> GraphState:
        field_key = state.get("current_field_key")
        if not field_key:
            return {"current_result": None}

        spec = FIELD_SPECS[field_key]
        candidates = state.get("current_candidates", [])
        try:
            result = extract_field_from_candidates(
                field_key=field_key,
                field_label=spec["label"],
                is_list=spec["is_list"],
                candidates=candidates,
            )
            return {"current_result": result}
        except Exception as exc:
            errors = list(state.get("errors", []))
            errors.append(f"extraction_failed:{field_key}:{type(exc).__name__}:{exc}")
            return {
                "current_result": empty_result_factory(spec["is_list"]),
                "errors": errors,
            }

    def store_field_result(state: GraphState) -> GraphState:
        field_key = state.get("current_field_key")
        if not field_key:
            return {"current_field_index": state["current_field_index"] + 1}

        results = dict(state.get("field_results", {}))
        current_result = state.get("current_result")
        if current_result is None:
            current_result = empty_result_factory(FIELD_SPECS[field_key]["is_list"])
        results[field_key] = current_result

        return {
            "field_results": results,
            "current_field_index": state["current_field_index"] + 1,
            "current_candidates": [],
            "current_result": None,
        }

    def finalize_report(state: GraphState) -> GraphState:
        from agent.schema import PropertyReportOutputModel

        report = PropertyReportOutputModel(
            file_name=state["file_name"],
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            model_used=state["model"],
            document_id=state.get("document_id"),
        )

        for field_key in state["field_keys"]:
            result = state.get("field_results", {}).get(field_key)
            if result is None:
                result = empty_result_factory(FIELD_SPECS[field_key]["is_list"])
            setattr(report, field_key, result)

        return {
            "generated_at": report.generated_at,
            "output": report,
        }

    def route_after_set_next_field(state: GraphState) -> str:
        if state.get("current_field_key") is None:
            return "finalize_report"
        return "retrieve_field_candidates"

    graph = StateGraph(GraphState)
    graph.add_node("init_report", init_report)
    graph.add_node("set_next_field", set_next_field)
    graph.add_node("retrieve_field_candidates", retrieve_field_candidates)
    graph.add_node("extract_field_value", extract_field_value)
    graph.add_node("store_field_result", store_field_result)
    graph.add_node("finalize_report", finalize_report)

    graph.add_edge(START, "init_report")
    graph.add_edge("init_report", "set_next_field")
    graph.add_conditional_edges(
        "set_next_field",
        route_after_set_next_field,
        {
            "retrieve_field_candidates": "retrieve_field_candidates",
            "finalize_report": "finalize_report",
        },
    )
    graph.add_edge("retrieve_field_candidates", "extract_field_value")
    graph.add_edge("extract_field_value", "store_field_result")
    graph.add_edge("store_field_result", "set_next_field")
    graph.add_edge("finalize_report", END)

    return graph.compile()
