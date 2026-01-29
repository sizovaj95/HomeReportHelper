import os
from pathlib import Path

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest


load_dotenv()

AZURE_LANG_ENDPOINT = os.getenv("AZURE_LANGUAGE_SERVICE_ENDPOINT", "")
AZURE_LANG_API_KEY = os.getenv("AZURE_LANGUAGE_SERVICE_API_KEY", "")


data_folder = Path(__file__).parent / "reports"
pdf_folder = data_folder / "pdf"
md_folder = data_folder / "md"
images_folder = data_folder / "images"


def main():
    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=AZURE_LANG_ENDPOINT, credential=AzureKeyCredential(AZURE_LANG_API_KEY)
    )

    with open(pdf_folder / "BertramHouseCarnwath.pdf", "rb") as f:
        doc_bytes = f.read()

    poller = document_intelligence_client.begin_analyze_document(
        model_id="prebuilt-layout",
        body=AnalyzeDocumentRequest(bytes_source=doc_bytes),
        output=["figures"]
    )
    result = poller.result()
    result

main()
