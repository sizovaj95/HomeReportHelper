from dataclasses import dataclass, field
from typing import Literal


SCHEMA_VERSION = "1.0"
BOUNDARY_SOURCE_AZURE_LAYOUT = "azure_layout_section"
PARAGRAPH_KIND_TEXT = "text"
PARAGRAPH_KIND_TABLE_TEXT = "table_text"
ParagraphKind = Literal["text", "table_text"]


@dataclass
class DocumentRecord:
    document_id: str
    schema_version: str
    file_name: str
    file_sha256: str
    page_count: int
    created_at: str


@dataclass
class SectionRecord:
    section_id: str
    document_id: str
    section_order: int
    title: str | None
    summary: str = ""
    boundary_source: str = BOUNDARY_SOURCE_AZURE_LAYOUT
    pages: list[int] = field(default_factory=list)
    paragraph_ids: list[str] = field(default_factory=list)
    merged_from_section_ids: list[str] = field(default_factory=list)
    is_heading_only_original: bool = False
    inherited_headings: list[str] = field(default_factory=list)


@dataclass
class ParagraphRecord:
    paragraph_id: str
    document_id: str
    section_id: str
    order_in_section: int
    kind: ParagraphKind
    text: str
    pages: list[int] = field(default_factory=list)
    layout_refs: list[str] = field(default_factory=list)
    role: str | None = None
    is_heading_like: bool = False
    merged_from_ids: list[str] = field(default_factory=list)
    embedding_model: str | None = None
    embedding_vector_id: str | None = None
    token_count: int | None = None
