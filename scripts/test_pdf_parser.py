from pathlib import Path

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.parsers.document_detector import DocumentDetector


def test_pdf(pdf_path: str):
    print("\n" + "=" * 80)
    print(f"Testing: {pdf_path}")
    print("=" * 80)

    parser = PDFParser()

    pages = parser.parse_pdf(pdf_path)

    print(f"\nTotal Pages Extracted: {len(pages)}")

    if not pages:
        print("❌ No pages extracted")
        return

    # First page preview
    print("\nFirst Page Preview:")
    print("-" * 80)
    print(pages[0]["text"][:1000])

    # First 3 pages for document detection
    first_pages_text = "\n".join(
        page["text"]
        for page in pages[:3]
    )

    try:
        doc_type = DocumentDetector.detect(
            first_pages_text
        )

        print(f"\n✅ Document Type Detected: {doc_type}")

    except Exception as e:
        print(f"\n❌ Detection Failed: {e}")


if __name__ == "__main__":

    pdf_files = [
        "data/raw/income_tax_act_1961.pdf",
        "data/raw/constitution_of_india.pdf",
    ]

    for pdf_file in pdf_files:

        if Path(pdf_file).exists():
            test_pdf(pdf_file)
        else:
            print(f"\n❌ File Not Found: {pdf_file}")