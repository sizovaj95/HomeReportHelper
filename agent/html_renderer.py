from __future__ import annotations

from html import escape

from agent.prompts import FIELD_SPECS
from agent.schema import ListFieldResultModel, PropertyReportOutputModel


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
    rows = []
    for field_key, spec in FIELD_SPECS.items():
        label = spec["label"]
        result = getattr(report, field_key)

        if isinstance(result, ListFieldResultModel):
            value_str = ", ".join(result.value) if result.value else "Not found"
        else:
            value_str = result.value if result.value else "Not found"

        row = (
            "<section class='field-card'>"
            f"<h3>{escape(label)}</h3>"
            f"<div><strong>Status:</strong> {escape(result.status)}</div>"
            f"<div><strong>Value:</strong> {escape(value_str)}</div>"
            f"{_render_pages(result.found_pages, result.candidate_pages)}"
            f"{_render_evidence(result)}"
            "</section>"
        )
        rows.append(row)

    return (
        "<!doctype html>"
        "<html><head><meta charset='utf-8'><title>Home Report Extraction</title>"
        "<style>"
        "body{font-family:Georgia,serif;max-width:980px;margin:24px auto;padding:0 16px;background:#f7f5ef;color:#1f1f1f;}"
        "h1{margin-bottom:6px;}"
        ".meta{margin-bottom:20px;color:#444;}"
        ".field-card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px 16px;margin:12px 0;}"
        "pre{white-space:pre-wrap;background:#f4f4f4;padding:8px;border-radius:6px;}"
        "details{margin-top:10px;}"
        "</style></head><body>"
        "<h1>Property Extraction Report</h1>"
        f"<div class='meta'><strong>Source file:</strong> {escape(report.file_name)}<br>"
        f"<strong>Generated at:</strong> {escape(report.generated_at)}<br>"
        f"<strong>Model:</strong> {escape(report.model_used)}</div>"
        + "".join(rows)
        + "</body></html>"
    )
