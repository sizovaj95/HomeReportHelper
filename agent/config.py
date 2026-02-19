from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "data_preparation" / "reports"
SQLITE_DB_PATH = str(BASE_DIR / "data_preparation" / "home_reports.db")
CHROMA_DIR = str(BASE_DIR / "data_preparation" / "chroma_db")
OUTPUT_DIR = BASE_DIR / "agent" / "output"

EXTRACTION_MODEL = "gpt-4.1-mini"
RETRIEVAL_TOP_K_VECTOR = 10
RETRIEVAL_TOP_K_KEYWORD = 15
FINAL_CANDIDATE_LIMIT = 12
