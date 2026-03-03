from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timezone

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
        self._ensure_query_embedding_cache_table()

    def get_section_overview(self, document_id: str) -> list[dict]:
        sections = self.storage.get_sections(document_id)
        return [
            {
                "section_id": section.section_id,
                "title": section.title or "",
                "summary": section.summary or "",
                "pages": section.pages,
            }
            for section in sections
        ]

    def prepare_unique_queries(self, query_hints: list[str]) -> list[str]:
        return self._prepare_multi_queries(query_hints)

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
        logger.info("Section keyword hits: %d", len(section_hits))
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

    def retrieve_candidates_from_sections(
        self,
        document_id: str,
        section_ids: list[str],
        query_hints: list[str],
        final_limit: int = config.FINAL_CANDIDATE_LIMIT,
    ) -> list[CandidateChunk]:
        if not section_ids:
            return []

        all_paragraphs = self.storage.get_paragraphs(document_id)
        selected = [
            p for p in all_paragraphs
            if p.section_id in set(section_ids) and not p.is_heading_like and p.text.strip()
        ]
        if not selected:
            return []

        normalized_terms = [t.strip().lower() for t in query_hints if t.strip()]
        candidates: list[CandidateChunk] = []
        for para in selected:
            text_lower = para.text.lower()
            keyword_hits = 0
            for term in normalized_terms:
                if term in text_lower:
                    keyword_hits += 1
                else:
                    term_tokens = [tok for tok in re.split(r"\\W+", term) if tok]
                    if term_tokens and all(tok in text_lower for tok in term_tokens):
                        keyword_hits += 1

            page = para.pages[0] if para.pages else None
            score = 0.6 + (0.1 * keyword_hits)
            candidates.append(
                CandidateChunk(
                    paragraph_id=para.paragraph_id,
                    section_id=para.section_id,
                    page=page,
                    text=para.text,
                    source="section_routed",
                    score=score,
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:final_limit]

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
        queries = self._prepare_multi_queries(query_hints)
        if not queries:
            return []

        try:
            import chromadb
        except ModuleNotFoundError:
            return []

        client = chromadb.PersistentClient(path=self.chroma_dir)
        collection = client.get_or_create_collection(name="home_report_paragraphs_v1")

        score_by_para_id: dict[str, float] = {}
        for query_idx, query_text in enumerate(queries):
            query_embedding = self._get_or_create_query_embedding(query_text)
            if query_embedding is None:
                continue

            try:
                response = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where={"document_id": document_id},
                    include=["metadatas", "distances"],
                )
            except Exception as exc:
                logger.warning(
                    "Vector query failed for query '%s', continuing with remaining queries: %s",
                    query_text,
                    exc,
                )
                continue

            metadatas = (response.get("metadatas") or [[]])[0]
            distances = (response.get("distances") or [[]])[0]
            query_weight = max(0.5, 1.0 - (query_idx * 0.08))

            for idx, meta in enumerate(metadatas):
                para_id = meta.get("paragraph_id")
                if not para_id:
                    continue
                distance = distances[idx] if idx < len(distances) else None
                if isinstance(distance, (float, int)):
                    sim_score = 1.0 / (1.0 + float(distance))
                else:
                    sim_score = 0.5
                combined = sim_score * query_weight
                score_by_para_id[para_id] = max(score_by_para_id.get(para_id, 0.0), combined)

        ranked_ids = [
            para_id
            for para_id, _ in sorted(score_by_para_id.items(), key=lambda item: item[1], reverse=True)
        ]
        return ranked_ids

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

    def _prepare_multi_queries(self, query_hints: list[str]) -> list[str]:
        seen: set[str] = set()
        queries: list[str] = []
        for hint in query_hints:
            normalized = self._normalize_query_text(hint)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            queries.append(normalized)
        return queries

    def _normalize_query_text(self, text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    def _get_or_create_query_embedding(self, query_text: str) -> list[float] | None:
        cached_db = self._load_query_embedding_from_db(query_text, OPENAI_EMBEDDING_MODEL)
        if cached_db is not None:
            return cached_db

        embedding = self._embed_query(query_text)
        if embedding is None:
            return None

        self._store_query_embedding_in_db(query_text, OPENAI_EMBEDDING_MODEL, embedding)
        return embedding

    def _ensure_query_embedding_cache_table(self) -> None:
        with sqlite3.connect(self.storage.sqlite_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_embedding_cache (
                    query_text TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (query_text, embedding_model)
                )
                """
            )

    def _load_query_embedding_from_db(self, query_text: str, model: str) -> list[float] | None:
        with sqlite3.connect(self.storage.sqlite_db_path) as conn:
            row = conn.execute(
                """
                SELECT embedding_json
                FROM query_embedding_cache
                WHERE query_text = ? AND embedding_model = ?
                LIMIT 1
                """,
                (query_text, model),
            ).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(row[0])
            if isinstance(value, list):
                return value
        except json.JSONDecodeError:
            return None
        return None

    def _store_query_embedding_in_db(self, query_text: str, model: str, embedding: list[float]) -> None:
        with sqlite3.connect(self.storage.sqlite_db_path) as conn:
            conn.execute(
                """
                INSERT INTO query_embedding_cache (query_text, embedding_model, embedding_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(query_text, embedding_model) DO UPDATE SET
                    embedding_json=excluded.embedding_json,
                    created_at=excluded.created_at
                """,
                (
                    query_text,
                    model,
                    json.dumps(embedding),
                    datetime.now(tz=timezone.utc).isoformat(),
                ),
            )
