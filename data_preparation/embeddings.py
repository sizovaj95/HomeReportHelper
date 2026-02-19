from __future__ import annotations

import asyncio
import logging
import os
from typing import Iterable

from dotenv import load_dotenv

from data_preparation import layout_objects as lo


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_EMBEDDING_MAX_CONCURRENCY = int(os.getenv("OPENAI_EMBEDDING_MAX_CONCURRENCY", "12"))
MIN_EMBEDDING_WORDS = int(os.getenv("MIN_EMBEDDING_WORDS", "5"))
OPENAI_EMBEDDING_MAX_RETRIES = int(os.getenv("OPENAI_EMBEDDING_MAX_RETRIES", "4"))
OPENAI_EMBEDDING_TIMEOUT_SECONDS = float(os.getenv("OPENAI_EMBEDDING_TIMEOUT_SECONDS", "60"))


logger = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(
        self,
        model: str = OPENAI_EMBEDDING_MODEL,
        max_concurrency: int = OPENAI_EMBEDDING_MAX_CONCURRENCY,
        max_retries: int = OPENAI_EMBEDDING_MAX_RETRIES,
        timeout_seconds: float = OPENAI_EMBEDDING_TIMEOUT_SECONDS,
    ):
        try:
            from openai import AsyncOpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for embeddings.") from exc

        if not OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY for embeddings.")

        self.model = model
        self.max_concurrency = max(1, max_concurrency)
        self.max_retries = max(1, max_retries)
        self.timeout_seconds = max(1.0, timeout_seconds)
        self.client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            timeout=self.timeout_seconds,
        )

    async def _embed_one(
        self,
        paragraph: lo.ParagraphRecord,
        semaphore: asyncio.Semaphore,
    ) -> tuple[str, list[float] | None, int]:
        if not self._should_embed(paragraph):
            return paragraph.paragraph_id, None, 0

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with semaphore:
                    response = await self.client.embeddings.create(
                        model=self.model,
                        input=paragraph.text,
                    )
                vector = response.data[0].embedding
                token_count = getattr(response.usage, "total_tokens", 0)
                return paragraph.paragraph_id, vector, token_count
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                # Exponential backoff with light jitter-free delay.
                await asyncio.sleep(0.75 * attempt)

        logger.warning(
            "Embedding failed after retries for paragraph_id=%s; skipping. Last error: %s",
            paragraph.paragraph_id,
            type(last_exc).__name__ if last_exc else "unknown",
        )
        return paragraph.paragraph_id, None, 0

    async def _embed_paragraphs_async(
        self,
        paragraphs: list[lo.ParagraphRecord],
    ) -> tuple[dict[str, list[float]], dict[str, int], str]:
        vectors: dict[str, list[float]] = {}
        token_counts: dict[str, int] = {}
        if not paragraphs:
            return vectors, token_counts, self.model

        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._embed_one(paragraph, semaphore) for paragraph in paragraphs]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        for paragraph_id, vector, token_count in results:
            if vector is None:
                continue
            vectors[paragraph_id] = vector
            token_counts[paragraph_id] = token_count

        if not vectors:
            logger.warning(
                "No embeddings were generated successfully for this document. "
                "Continuing without vector updates."
            )

        return vectors, token_counts, self.model

    def embed_paragraphs(
        self,
        paragraphs: list[lo.ParagraphRecord],
    ) -> tuple[dict[str, list[float]], dict[str, int], str]:
        return asyncio.run(self._embed_paragraphs_async(paragraphs))

    def _should_embed(self, paragraph: lo.ParagraphRecord) -> bool:
        if paragraph.is_heading_like:
            return False

        text = paragraph.text.strip()
        if not text:
            return False

        word_count = len(text.split())
        return word_count >= MIN_EMBEDDING_WORDS
