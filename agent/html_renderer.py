from __future__ import annotations

from collections import defaultdict
from html import escape

from agent.prompts import FIELD_GROUP_ORDER, FIELD_SPECS, FieldGroup
from agent.schema import PropertyReportOutputModel

BULLET_FORMAT_FIELDS = {
    "recommended_efficiency_measures",
    "potential_problems",
    "additional_costs",
    "renovations_done_and_when",
}


def _render_field_card(field_key: str, label: str, result) -> str:
    value_text = result.value if result.value else "Not found"
    if field_key in BULLET_FORMAT_FIELDS and value_text != "Not found":
        bullet_lines = [line.strip() for line in value_text.splitlines() if line.strip()]
        bullet_items = [line[2:].strip() if line.startswith("- ") else line for line in bullet_lines]
        value_block = "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in bullet_items) + "</ul>"
    else:
        value_block = escape(value_text)
    return (
        "<section class='field-card'>"
        f"<h3>{escape(label)}</h3>"
        f"<div><strong>Status:</strong> {escape(result.status)}</div>"
        f"<div><strong>Value:</strong> {value_block}</div>"
        f"{_render_pages(result.found_pages, result.candidate_pages)}"
        f"{_render_evidence(result)}"
        "</section>"
    )


def _group_field_specs() -> list[tuple[str, list[tuple[str, dict]]]]:
    grouped: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for field_key, spec in FIELD_SPECS.items():
        group = spec.get("ui_group", FieldGroup.PROPERTY_OVERVIEW)
        group_label = str(group)
        grouped[group_label].append((field_key, spec))

    for group_label in grouped:
        grouped[group_label].sort(key=lambda item: item[1].get("ui_order", 9999))

    ordered_groups: list[tuple[str, list[tuple[str, dict]]]] = []
    for group in FIELD_GROUP_ORDER:
        group_label = str(group)
        if group_label in grouped:
            ordered_groups.append((group_label, grouped.pop(group_label)))

    for group_label in sorted(grouped.keys()):
        ordered_groups.append((group_label, grouped[group_label]))

    return ordered_groups


def _render_pages(found_pages: list[int], candidate_pages: list[int]) -> str:
    found = ", ".join(str(p) for p in found_pages) if found_pages else "None"
    candidate = ", ".join(str(p) for p in candidate_pages) if candidate_pages else "None"
    return f"<div><strong>Found pages:</strong> {escape(found)}</div><div><strong>Candidate pages:</strong> {escape(candidate)}</div>"


def _render_evidence(result) -> str:
    if not result.evidence_paragraphs:
        return "<details><summary>Evidence paragraphs</summary><p>No direct evidence paragraphs captured.</p></details>"

    items = []
    for ev in result.evidence_paragraphs:
        page_label = f"Page {ev.page}" if ev.page is not None else "Page N/A"
        items.append(
            "<div class='evidence-item'>"
            f"<div><strong>{escape(page_label)}</strong> | paragraph_id={escape(ev.paragraph_id)}</div>"
            f"<pre>{escape(ev.text)}</pre>"
            "</div>"
        )
    return "<details><summary>Evidence paragraphs</summary>" + "".join(items) + "</details>"


def render_html(report: PropertyReportOutputModel) -> str:
    groups_html: list[str] = []
    for group_label, fields in _group_field_specs():
        field_cards: list[str] = []
        for field_key, spec in fields:
            result = getattr(report, field_key)
            field_cards.append(_render_field_card(field_key, spec["label"], result))
        groups_html.append(
            "<section class='group-card'>"
            f"<h2>{escape(group_label)}</h2>"
            + "".join(field_cards)
            + "</section>"
        )

    return (
        "<!doctype html>"
        "<html><head><meta charset='utf-8'><title>Home Report Extraction</title>"
        "<style>"
        "body{font-family:Georgia,serif;max-width:980px;margin:24px auto;padding:0 16px;background:#f7f5ef;color:#1f1f1f;}"
        "h1{margin-bottom:6px;}"
        ".meta{margin-bottom:20px;color:#444;}"
        ".group-card{margin:18px 0 22px;}"
        ".group-card h2{margin:12px 0 8px;padding-bottom:4px;border-bottom:2px solid #ddd;}"
        ".field-card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px 16px;margin:12px 0;}"
        "pre{white-space:pre-wrap;background:#f4f4f4;padding:8px;border-radius:6px;}"
        "details{margin-top:10px;}"
        "</style></head><body>"
        "<h1>Property Extraction Report</h1>"
        f"<div class='meta'><strong>Source file:</strong> {escape(report.file_name)}<br>"
        f"<strong>Generated at:</strong> {escape(report.generated_at)}<br>"
        f"<strong>Model:</strong> {escape(report.model_used)}</div>"
        + "".join(groups_html)
        + "</body></html>"
    )
