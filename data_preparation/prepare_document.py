import argparse
import json
from pathlib import Path

from data_preparation.pipeline_service import prepare_document_if_needed


DATA_PREPARATION_DIR = Path(__file__).resolve().parent
DEFAULT_REPORTS_DIR = DATA_PREPARATION_DIR / "reports" / "pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OCR layout into canonical records.")
    parser.add_argument("--pdf-path", default=None, help="Path to source PDF")
    parser.add_argument("--skip-summaries", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--sqlite-db", default="data_preparation/home_reports.db")
    parser.add_argument("--chroma-dir", default="data_preparation/chroma_db")
    return parser.parse_args()


def choose_pdf_from_reports_dir(reports_dir: Path) -> Path:
    if not reports_dir.exists() or not reports_dir.is_dir():
        raise FileNotFoundError(f"Reports folder not found: {reports_dir}")

    pdf_files = sorted(reports_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in reports folder: {reports_dir}")

    print("Available reports:")
    for idx, pdf in enumerate(pdf_files, start=1):
        print(f"{idx}. {pdf.name}")

    while True:
        choice = input("Pick a report by number: ").strip()
        if not choice.isdigit():
            print("Please enter a number.")
            continue

        number = int(choice)
        if number < 1 or number > len(pdf_files):
            print(f"Please choose a number between 1 and {len(pdf_files)}.")
            continue
        return pdf_files[number - 1]


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf_path) if args.pdf_path else choose_pdf_from_reports_dir(DEFAULT_REPORTS_DIR)

    result = prepare_document_if_needed(
        pdf_path=pdf_path,
        sqlite_db=args.sqlite_db,
        chroma_dir=args.chroma_dir,
        run_summaries=not args.skip_summaries,
        run_embeddings=not args.skip_embeddings,
    )

    print(
        json.dumps(
            {
                "document_id": result.document_id,
                "file_name": result.file_name,
                "file_sha256": result.file_sha256,
                "was_prepared_now": result.was_prepared_now,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
