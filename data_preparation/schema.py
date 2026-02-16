from pydantic import BaseModel, Field


class DocumentRecordModel(BaseModel):
    document_id: str
    schema_version: str
    file_name: str
    file_sha256: str
    page_count: int = Field(ge=0)
    created_at: str


class SectionRecordModel(BaseModel):
    section_id: str
    document_id: str
    section_order: int = Field(ge=1)
    title: str | None = None
    summary: str = ""
    boundary_source: str
    pages: list[int] = Field(default_factory=list)
    paragraph_ids: list[str] = Field(default_factory=list)
    merged_from_section_ids: list[str] = Field(default_factory=list)
    is_heading_only_original: bool = False
    inherited_headings: list[str] = Field(default_factory=list)


class ParagraphRecordModel(BaseModel):
    paragraph_id: str
    document_id: str
    section_id: str
    order_in_section: int = Field(ge=1)
    kind: str
    text: str
    pages: list[int] = Field(default_factory=list)
    layout_refs: list[str] = Field(default_factory=list)
    role: str | None = None
    is_heading_like: bool = False
    merged_from_ids: list[str] = Field(default_factory=list)
    embedding_model: str | None = None
    embedding_vector_id: str | None = None
    token_count: int | None = None
