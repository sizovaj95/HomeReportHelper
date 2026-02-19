from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from agent import config
from agent.models import CandidateChunk
from agent.storage import AgentStorage, StoredParagraph


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, storage: AgentStorage, chroma_dir: str = config.CHROMA_DIR):
        self.storage = storage
        self.chroma_dir = chroma_dir
        self._openai_client = self._init_openai_client()

    def retrieve_candidates(
        self,
        document_id: str,
        query_hints: list[str],
        top_k_vector: int = config.RETRIEVAL_TOP_K_VECTOR,
        top_k_keyword: int = config.RETRIEVAL_TOP_K_KEYWORD,
        final_limit: int = config.FINAL_CANDIDATE_LIMIT,
    ) -> list[CandidateChunk]:
        all_paragraphs = self.storage.get_paragraphs(document_id)
        paragraph_map = {p.paragraph_id: p for p in all_paragraphs}

        scored: dict[str, CandidateChunk] = {}

        # Keyword results from paragraph text.
        keyword_hits = self.storage.keyword_search_paragraphs(document_id, query_hints, top_k_keyword)
        for idx, para in enumerate(keyword_hits):
            self._add_or_update(scored, para, source="keyword", score=1.0 - (idx * 0.02))

        # Keyword results from section summaries -> include paragraphs in those sections.
        section_hits = self.storage.keyword_search_sections(document_id, query_hints, max(5, top_k_keyword // 3))
        for section in section_hits:
            for para in all_paragraphs:
                if para.section_id == section.section_id:
                    self._add_or_update(scored, para, source="section_summary", score=0.45)

        # Vector results from Chroma.
        vector_hits = self._vector_search(document_id, query_hints, top_k_vector)
        for idx, para_id in enumerate(vector_hits):
            para = paragraph_map.get(para_id)
            if para is None:
                continue
            self._add_or_update(scored, para, source="vector", score=1.1 - (idx * 0.03))

        ranked = sorted(scored.values(), key=lambda c: c.score, reverse=True)
        return ranked[:final_limit]

    def _add_or_update(self, scored: dict[str, CandidateChunk], para: StoredParagraph, source: str, score: float) -> None:
        page = para.pages[0] if para.pages else None
        existing = scored.get(para.paragraph_id)
        if existing is None or score > existing.score:
            scored[para.paragraph_id] = CandidateChunk(
                paragraph_id=para.paragraph_id,
                section_id=para.section_id,
                page=page,
                text=para.text,
                source=source,
                score=score,
            )

    def _vector_search(self, document_id: str, query_hints: list[str], top_k: int) -> list[str]:
        query_text = " | ".join(query_hints).strip()
        if not query_text:
            return []

        try:
            import chromadb
        except ModuleNotFoundError:
            return []

        query_embedding = self._embed_query(query_text)
        if query_embedding is None:
            return []

        client = chromadb.PersistentClient(path=self.chroma_dir)
        collection = client.get_or_create_collection(name="home_report_paragraphs_v1")

        try:
            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"document_id": document_id},
                include=["metadatas"],
            )
        except Exception as exc:
            logger.warning("Vector query failed, continuing with keyword retrieval only: %s", exc)
            return []

        ids: list[str] = []
        metadatas = response.get("metadatas", [])
        for meta_row in metadatas:
            for meta in meta_row:
                para_id = meta.get("paragraph_id")
                if para_id:
                    ids.append(para_id)

        # Keep stable uniqueness order.
        seen = set()
        unique_ids = []
        for para_id in ids:
            if para_id in seen:
                continue
            seen.add(para_id)
            unique_ids.append(para_id)
        return unique_ids

    def _init_openai_client(self):
        if not OPENAI_API_KEY:
            return None
        try:
            from openai import OpenAI
        except ModuleNotFoundError:
            return None
        return OpenAI(api_key=OPENAI_API_KEY)

    def _embed_query(self, query_text: str) -> list[float] | None:
        if self._openai_client is None:
            logger.warning("OPENAI_API_KEY or openai package missing; vector retrieval disabled.")
            return None
        try:
            response = self._openai_client.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=query_text,
            )
            return response.data[0].embedding
        except Exception as exc:
            logger.warning("Failed to create query embedding, vector retrieval disabled for this field: %s", exc)
            return None
