from typing import Literal

from pydantic import BaseModel, Field


FieldStatus = Literal["found", "not_found", "ambiguous"]


class EvidenceParagraph(BaseModel):
    paragraph_id: str
    page: int | None = None
    text: str
    relevance_note: str | None = None


class FieldResultModel(BaseModel):
    value: str | None = None
    status: FieldStatus = "not_found"
    found_pages: list[int] = Field(default_factory=list)
    candidate_pages: list[int] = Field(default_factory=list)
    evidence_paragraphs: list[EvidenceParagraph] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

class PropertyReportOutputModel(BaseModel):
    schema_version: str = "1.0"
    file_name: str
    generated_at: str
    model_used: str
    document_id: str | None = None

    property_address: FieldResultModel = Field(default_factory=FieldResultModel)
    property_age: FieldResultModel = Field(default_factory=FieldResultModel)
    property_area: FieldResultModel = Field(default_factory=FieldResultModel)
    property_epc: FieldResultModel = Field(default_factory=FieldResultModel)
    council_tax_code: FieldResultModel = Field(default_factory=FieldResultModel)
    recommended_efficiency_measures: FieldResultModel = Field(default_factory=FieldResultModel)
    window_glazing: FieldResultModel = Field(default_factory=FieldResultModel)
    external_walls_material: FieldResultModel = Field(default_factory=FieldResultModel)
    internal_walls_material: FieldResultModel = Field(default_factory=FieldResultModel)
    gas_and_boiler_notes: FieldResultModel = Field(default_factory=FieldResultModel)
    electricity_notes: FieldResultModel = Field(default_factory=FieldResultModel)
    potential_problems: FieldResultModel = Field(default_factory=FieldResultModel)
    additional_costs: FieldResultModel = Field(default_factory=FieldResultModel)
    special_building_notes: FieldResultModel = Field(default_factory=FieldResultModel)
    market_value: FieldResultModel = Field(default_factory=FieldResultModel)


class HtmlAgentOutputModel(BaseModel):
    extracted_data: PropertyReportOutputModel
    html: str
