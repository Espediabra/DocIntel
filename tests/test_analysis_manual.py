from pathlib import Path

from services.document_service import DocumentService
from services.analysis_service import AnalysisService


PDF_PATH = Path("data/test.pdf")


def test_analysis():
    document_service = DocumentService()
    document = document_service.extract(str(PDF_PATH))

    analysis_service = AnalysisService()
    result = analysis_service.analyze(document)

    print("\n" + "=" * 80)
    print("ANALYSE DU DOCUMENT")
    print("=" * 80)
    print(result)
    print("=" * 80)

    assert result
    assert len(result.strip()) > 0