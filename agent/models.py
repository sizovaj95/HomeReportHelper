from dataclasses import dataclass


@dataclass
class CandidateChunk:
    paragraph_id: str
    section_id: str
    page: int | None
    text: str
    source: str
    score: float
