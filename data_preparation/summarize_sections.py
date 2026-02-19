from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from data_preparation import layout_objects as lo


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_SUMMARY_MAX_CONCURRENCY = int(os.getenv("OPENAI_SUMMARY_MAX_CONCURRENCY", "8"))


SYSTEM_PROMPT = (
    "You write a strict section description for OCR text from property home reports. "
    "Return exactly one short sentence describing what the text is about "
    "(for example: potential savings, condition notes, repairs, energy performance). "
    "Use only information explicitly present in the text. "
    "Do not infer, do not speculate, do not add facts, numbers, causes, or recommendations "
    "that are not directly stated. "
    "No bullet points and no extra formatting."
)


class SectionSummarizer:
    def __init__(
        self,
        model: str = OPENAI_CHAT_MODEL,
        max_retries: int = 3,
        max_concurrency: int = OPENAI_SUMMARY_MAX_CONCURRENCY,
    ):
        try:
            from openai import AsyncOpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for summaries.") from exc

        if not OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY for section summaries.")

        self.model = model
        self.max_retries = max_retries
        self.max_concurrency = max(1, max_concurrency)
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def summarize(self, section_text: str) -> str:
        content = section_text.strip()
        if not content:
            return ""

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                "Write one strict description sentence for this section text. "
                                "Do not include any information not explicitly present.\n\n"
                                f"{content}"
                            ),
                        },
                    ],
                )
                summary = response.choices[0].message.content or ""
                return summary.strip()
            except Exception:
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(0.6 * attempt)

        return ""

    async def _summarize_sections_async(
        self,
        sections: list[lo.SectionRecord],
        section_text_map: dict[str, str],
    ) -> list[str]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _summarize_one(section: lo.SectionRecord) -> str:
            async with semaphore:
                return await self.summarize(section_text_map.get(section.section_id, ""))

        tasks = [_summarize_one(section) for section in sections]
        return await asyncio.gather(*tasks)

    def summarize_sections(
        self,
        sections: list[lo.SectionRecord],
        section_text_map: dict[str, str],
    ) -> None:
        summaries = asyncio.run(self._summarize_sections_async(sections, section_text_map))
        for section, summary in zip(sections, summaries):
            section.summary = summary
