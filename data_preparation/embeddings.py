from __future__ import annotations

import os
from typing import Iterable

from dotenv import load_dotenv

import layout_objects as lo


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")


class EmbeddingClient:
    def __init__(self, model: str = OPENAI_EMBEDDING_MODEL):
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for embeddings.") from exc

        if not OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY for embeddings.")

        self.model = model
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def embed_paragraphs(
        self,
        paragraphs: Iterable[lo.ParagraphRecord],
    ) -> tuple[dict[str, list[float]], dict[str, int], str]:
        vectors: dict[str, list[float]] = {}
        token_counts: dict[str, int] = {}

        paragraph_list = list(paragraphs)
        if not paragraph_list:
            return vectors, token_counts, self.model

        for paragraph in paragraph_list:
            if not paragraph.text.strip():
                continue
            response = self.client.embeddings.create(
                model=self.model,
                input=paragraph.text,
            )
            vectors[paragraph.paragraph_id] = response.data[0].embedding
            token_counts[paragraph.paragraph_id] = getattr(response.usage, "total_tokens", 0)

        return vectors, token_counts, self.model
