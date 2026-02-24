from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "data_preparation" / "reports"
SQLITE_DB_PATH = str(BASE_DIR / "data_preparation" / "home_reports.db")
CHROMA_DIR = str(BASE_DIR / "data_preparation" / "chroma_db")
OUTPUT_DIR = BASE_DIR / "agent" / "output"

EXTRACTION_MODEL = "gpt-4.1-mini"
SECTION_MODEL = "gpt-4o-mini"
RETRIEVAL_TOP_K_VECTOR = 10
RETRIEVAL_TOP_K_KEYWORD = 15
# Dynamic evidence budget for per-field extraction prompts.
FINAL_CANDIDATE_LIMIT_BASE = 8
FINAL_CANDIDATE_LIMIT_PER_QUERY = 2
FINAL_CANDIDATE_LIMIT_MAX = 24
RETRIEVAL_KEYWORD_BUFFER = 4
# Legacy fallback constant kept for compatibility in call sites that still use defaults.
FINAL_CANDIDATE_LIMIT = 12
