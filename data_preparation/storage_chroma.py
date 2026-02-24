from __future__ import annotations

from pathlib import Path

from data_preparation import layout_objects as lo


class ChromaStore:
    def __init__(self, persist_directory: str = "data_preparation/chroma_db"):
        try:
            import chromadb
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "chromadb is required for embedding persistence. Install chromadb."
            ) from exc

        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="home_report_paragraphs_v1"
        )

    @staticmethod
    def paragraph_vector_id(document_id: str, paragraph_id: str) -> str:
        return f"{document_id}:{paragraph_id}"

    @staticmethod
    def section_key(document_id: str, section_id: str) -> str:
        return f"{document_id}:{section_id}"

    def upsert_paragraph_vectors(
        self,
        paragraphs: list[lo.ParagraphRecord],
        vectors_by_paragraph_id: dict[str, list[float]],
        section_order_by_id: dict[str, int],
    ) -> dict[str, str]:
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        embeddings: list[list[float]] = []
        vector_id_map: dict[str, str] = {}

        for paragraph in paragraphs:
            vector = vectors_by_paragraph_id.get(paragraph.paragraph_id)
            if vector is None:
                continue

            vector_id = self.paragraph_vector_id(paragraph.document_id, paragraph.paragraph_id)
            ids.append(vector_id)
            vector_id_map[paragraph.paragraph_id] = vector_id
            documents.append(paragraph.text)
            embeddings.append(vector)
            metadatas.append(
                {
                    "document_id": paragraph.document_id,
                    "section_id": paragraph.section_id,
                    "section_key": self.section_key(paragraph.document_id, paragraph.section_id),
                    "paragraph_id": paragraph.paragraph_id,
                    "paragraph_key": vector_id,
                    "kind": paragraph.kind,
                    "pages": ",".join(str(p) for p in paragraph.pages),
                    "section_order": section_order_by_id.get(paragraph.section_id, 0),
                    "paragraph_order": paragraph.order_in_section,
                }
            )

        if ids:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

        return vector_id_map
