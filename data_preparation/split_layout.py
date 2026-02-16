import re
from collections import defaultdict

from azure.ai.documentintelligence.models import AnalyzeResult, DocumentTable

import layout_objects as lo


ID_PATTERN = r"\/([a-z]{1,15})\/(\d{1,5})$"
PARAGRAPHS = "paragraphs"
TABLES = "tables"
SECTIONS = "sections"

HEADING_ROLES = {"title", "heading"}
SHORT_CHAR_THRESHOLD = 40
SHORT_WORD_THRESHOLD = 8


class LayoutElementId:
    """Parses Azure layout element identifiers like /paragraphs/12."""

    def __init__(self, element_id: str):
        if match := re.search(ID_PATTERN, element_id):
            self.type = match.group(1)
            self.id_as_num = int(match.group(2))
        else:
            raise ValueError(f"Wrong id format: {element_id}")


class LayoutProcessor:
    REQUIRED_ELEMENTS = {PARAGRAPHS, TABLES}

    def __init__(self, layout: AnalyzeResult, document_id: str):
        self.layout = layout
        self.document_id = document_id
        self.sections = layout.sections or []
        self.paragraphs = layout.paragraphs or []
        self.tables = layout.tables or []

        self.section_records: list[lo.SectionRecord] = []
        self.paragraph_records: list[lo.ParagraphRecord] = []

    def process(self) -> tuple[list[lo.SectionRecord], list[lo.ParagraphRecord]]:
        self.build_raw_records()
        self.merge_short_paragraphs_within_section()
        self.merge_heading_only_sections()
        self.finalize_ordering()
        return self.section_records, self.paragraph_records

    def build_raw_records(self) -> None:
        paragraph_counter = 1

        for section_idx, section in enumerate(self.sections):
            section_id = f"sec_{section_idx + 1:04d}"
            section_record = lo.SectionRecord(
                section_id=section_id,
                document_id=self.document_id,
                section_order=section_idx + 1,
                title=None,
                summary="",
            )

            section_elements = section.elements or []
            for element_ref in section_elements:
                try:
                    parsed_ref = LayoutElementId(element_ref)
                except ValueError:
                    continue

                if parsed_ref.type not in self.REQUIRED_ELEMENTS:
                    continue

                if parsed_ref.type == PARAGRAPHS:
                    paragraph_record = self._paragraph_record_from_layout(
                        paragraph_idx=parsed_ref.id_as_num,
                        paragraph_id=f"par_{paragraph_counter:05d}",
                        section_id=section_id,
                        order_in_section=len(section_record.paragraph_ids) + 1,
                        layout_ref=element_ref,
                    )
                else:
                    paragraph_record = self._paragraph_record_from_table(
                        table_idx=parsed_ref.id_as_num,
                        paragraph_id=f"par_{paragraph_counter:05d}",
                        section_id=section_id,
                        order_in_section=len(section_record.paragraph_ids) + 1,
                        layout_ref=element_ref,
                    )

                if paragraph_record is None:
                    continue

                section_record.paragraph_ids.append(paragraph_record.paragraph_id)
                section_record.pages = sorted(set(section_record.pages + paragraph_record.pages))
                self.paragraph_records.append(paragraph_record)

                if section_record.title is None and paragraph_record.is_heading_like:
                    section_record.title = paragraph_record.text

                paragraph_counter += 1

            self.section_records.append(section_record)

    def _paragraph_record_from_layout(
        self,
        paragraph_idx: int,
        paragraph_id: str,
        section_id: str,
        order_in_section: int,
        layout_ref: str,
    ) -> lo.ParagraphRecord | None:
        if paragraph_idx >= len(self.paragraphs):
            return None

        paragraph = self.paragraphs[paragraph_idx]
        text = (paragraph.content or "").strip()
        role = paragraph.role
        is_heading_like = role in HEADING_ROLES if role else self._fallback_heading_like(text)

        return lo.ParagraphRecord(
            paragraph_id=paragraph_id,
            document_id=self.document_id,
            section_id=section_id,
            order_in_section=order_in_section,
            kind=lo.PARAGRAPH_KIND_TEXT,
            text=text,
            pages=self._extract_pages(getattr(paragraph, "bounding_regions", None)),
            layout_refs=[layout_ref],
            role=role,
            is_heading_like=is_heading_like,
            merged_from_ids=[paragraph_id],
        )

    def _paragraph_record_from_table(
        self,
        table_idx: int,
        paragraph_id: str,
        section_id: str,
        order_in_section: int,
        layout_ref: str,
    ) -> lo.ParagraphRecord | None:
        if table_idx >= len(self.tables):
            return None

        table = self.tables[table_idx]
        table_text = self.flatten_table_to_text(table)
        return lo.ParagraphRecord(
            paragraph_id=paragraph_id,
            document_id=self.document_id,
            section_id=section_id,
            order_in_section=order_in_section,
            kind=lo.PARAGRAPH_KIND_TABLE_TEXT,
            text=table_text,
            pages=self._extract_pages(getattr(table, "bounding_regions", None)),
            layout_refs=[layout_ref],
            role=None,
            is_heading_like=False,
            merged_from_ids=[paragraph_id],
        )

    def flatten_table_to_text(self, table: DocumentTable) -> str:
        rows: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for cell in table.cells:
            row_idx = cell.row_index if cell.row_index is not None else 0
            col_idx = cell.column_index if cell.column_index is not None else 0
            rows[row_idx].append((col_idx, (cell.content or "").strip()))

        lines: list[str] = []
        for row_idx in sorted(rows.keys()):
            ordered_cells = [cell_text for _, cell_text in sorted(rows[row_idx], key=lambda x: x[0])]
            lines.append(" | ".join(ordered_cells))

        return "\n".join(lines).strip()

    def merge_short_paragraphs_within_section(self) -> None:
        para_map = {p.paragraph_id: p for p in self.paragraph_records}

        for section in self.section_records:
            changed = True
            while changed:
                changed = False
                for idx, paragraph_id in enumerate(list(section.paragraph_ids)):
                    paragraph = para_map.get(paragraph_id)
                    if paragraph is None:
                        continue

                    if not self._is_merge_candidate(paragraph):
                        continue

                    target_idx = self._find_merge_target_index(section, idx, para_map)
                    if target_idx is None:
                        continue

                    target_id = section.paragraph_ids[target_idx]
                    target = para_map[target_id]

                    target.text = self._join_text(target.text, paragraph.text)
                    target.pages = sorted(set(target.pages + paragraph.pages))
                    target.layout_refs.extend(paragraph.layout_refs)
                    target.merged_from_ids.extend(paragraph.merged_from_ids)

                    section.paragraph_ids.pop(idx)
                    del para_map[paragraph_id]
                    changed = True
                    break

        self.paragraph_records = [p for p in self.paragraph_records if p.paragraph_id in para_map]

    def merge_heading_only_sections(self) -> None:
        if not self.section_records:
            return

        para_map = {p.paragraph_id: p for p in self.paragraph_records}
        idx = 0
        while idx < len(self.section_records):
            section = self.section_records[idx]
            body_paragraphs = [
                para_map[pid]
                for pid in section.paragraph_ids
                if pid in para_map and not para_map[pid].is_heading_like
            ]

            if body_paragraphs:
                idx += 1
                continue

            section.is_heading_only_original = True

            if len(self.section_records) == 1:
                idx += 1
                continue

            target_idx = idx + 1 if idx + 1 < len(self.section_records) else idx - 1
            target = self.section_records[target_idx]

            heading_texts = [
                para_map[pid].text
                for pid in section.paragraph_ids
                if pid in para_map and para_map[pid].is_heading_like and para_map[pid].text
            ]

            target.merged_from_section_ids.append(section.section_id)
            target.inherited_headings.extend(heading_texts)

            # Move all paragraphs to target section and keep document order.
            if target_idx > idx:
                target.paragraph_ids = section.paragraph_ids + target.paragraph_ids
            else:
                target.paragraph_ids.extend(section.paragraph_ids)

            for paragraph_id in section.paragraph_ids:
                if paragraph_id in para_map:
                    para_map[paragraph_id].section_id = target.section_id

            target.pages = sorted(set(target.pages + section.pages))
            self.section_records.pop(idx)
            if target_idx < idx:
                idx -= 1

        self.paragraph_records = list(para_map.values())

    def finalize_ordering(self) -> None:
        para_map = {p.paragraph_id: p for p in self.paragraph_records}
        ordered_paragraphs: list[lo.ParagraphRecord] = []

        for section_idx, section in enumerate(self.section_records):
            section.section_order = section_idx + 1

            clean_ids: list[str] = []
            for para_idx, paragraph_id in enumerate(section.paragraph_ids):
                paragraph = para_map.get(paragraph_id)
                if paragraph is None:
                    continue
                paragraph.order_in_section = para_idx + 1
                paragraph.section_id = section.section_id
                clean_ids.append(paragraph_id)
                ordered_paragraphs.append(paragraph)

            section.paragraph_ids = clean_ids
            section.pages = sorted(
                {
                    page
                    for paragraph_id in clean_ids
                    for page in para_map[paragraph_id].pages
                }
            )
            if section.title is None:
                for paragraph_id in clean_ids:
                    paragraph = para_map[paragraph_id]
                    if paragraph.is_heading_like and paragraph.text:
                        section.title = paragraph.text
                        break

        self.paragraph_records = ordered_paragraphs

    def build_section_text_for_summary(self, section: lo.SectionRecord) -> str:
        para_map = {p.paragraph_id: p for p in self.paragraph_records}
        chunks: list[str] = []

        if section.inherited_headings:
            chunks.append("Inherited headings:\n" + "\n".join(section.inherited_headings))

        body_text = []
        for paragraph_id in section.paragraph_ids:
            paragraph = para_map.get(paragraph_id)
            if paragraph is None:
                continue
            if paragraph.is_heading_like and paragraph.kind == lo.PARAGRAPH_KIND_TEXT:
                continue
            if paragraph.text:
                body_text.append(paragraph.text)

        if body_text:
            chunks.append("\n\n".join(body_text))

        return "\n\n".join(chunks).strip()

    def _extract_pages(self, bounding_regions) -> list[int]:
        if not bounding_regions:
            return []
        pages = [getattr(region, "page_number", None) for region in bounding_regions]
        return sorted({page for page in pages if page is not None})

    def _fallback_heading_like(self, text: str) -> bool:
        if not text:
            return False
        words = text.split()
        if len(words) > 12:
            return False
        return text.isupper()

    def _is_merge_candidate(self, paragraph: lo.ParagraphRecord) -> bool:
        if paragraph.kind != lo.PARAGRAPH_KIND_TEXT:
            return False
        if paragraph.is_heading_like:
            return False

        text = (paragraph.text or "").strip()
        if not text:
            return True

        return len(text) < SHORT_CHAR_THRESHOLD or len(text.split()) < SHORT_WORD_THRESHOLD

    def _find_merge_target_index(
        self,
        section: lo.SectionRecord,
        source_idx: int,
        para_map: dict[str, lo.ParagraphRecord],
    ) -> int | None:
        # Prefer previous compatible neighbor, otherwise next.
        previous_idx = source_idx - 1
        if previous_idx >= 0:
            previous_para = para_map.get(section.paragraph_ids[previous_idx])
            if previous_para and self._is_compatible_merge_target(previous_para):
                return previous_idx

        next_idx = source_idx + 1
        if next_idx < len(section.paragraph_ids):
            next_para = para_map.get(section.paragraph_ids[next_idx])
            if next_para and self._is_compatible_merge_target(next_para):
                return next_idx

        return None

    def _is_compatible_merge_target(self, paragraph: lo.ParagraphRecord) -> bool:
        return paragraph.kind == lo.PARAGRAPH_KIND_TEXT and not paragraph.is_heading_like

    def _join_text(self, base_text: str, append_text: str) -> str:
        if not base_text:
            return append_text
        if not append_text:
            return base_text
        return f"{base_text}\n{append_text}"
