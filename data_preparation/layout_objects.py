from dataclasses import dataclass

import pandas as pd


@dataclass
class BaseSectionObject:
    page: int
    content: str
    id: str


@dataclass
class Section:
    parts: list
    short_description: str  # ?
    title: str
    id: str


@dataclass
class Paragraph:
    text: str
    page: int
    id: str
    embedding: list[float]


@dataclass
class Table:
    id: str
    short_description: str
    page: int
    content: str | pd.DataFrame
