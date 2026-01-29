import os
from pathlib import Path
import pickle

from azure.ai.documentintelligence.models._models import AnalyzeResult, DocumentTable
import pandas as pd

import layout_objects as lo


def convert_table_with_header_to_pandas(table: DocumentTable) -> pd.DataFrame:
    table_as_lists = []
    current_row = 0
    while True:
        row = [cell for cell in table.cells if cell.row_index == current_row]
        if not row:
            break
        table_as_lists.append([cell.content for cell in row])
        current_row += 1
    df = pd.DataFrame(table_as_lists)
    return df


def convert_table_with_no_header_to_text(table: DocumentTable) -> str:
    table_as_text = []
    current_row = 0
    line = []
    for cell in table.cells:
        row_index = cell.row_index
        if row_index == current_row:
            line.append(cell.content)
        else:
            table_as_text.append(" ".join(line))
            line = [cell.content]
            current_row = row_index
    table_as_text.append(" ".join(line))
    return '\n'.join(table_as_text)


def document_layout_to_structured_format(layout: AnalyzeResult):
    sections = layout.sections
    paragrpaphs = layout.paragraphs
    tables = layout.tables
    figures = layout.figures
    no_header_tables = []
    header_tables = []
    no_header_tables_map = {}
    header_tables_map = {}
    for i, table in enumerate(tables):
        if table.cells[0].kind is None:
            no_header_tables_map[i] = convert_table_with_no_header_to_text(table)
        else:
            header_tables_map[i] = convert_table_with_header_to_pandas(table)
    return


if __name__ == "__main__":
    with open("example_layout.pkl", "rb") as f:
        example_layout = pickle.load(f)
    document_layout_to_structured_format(example_layout)
