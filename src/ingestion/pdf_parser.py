"""Parse legal PDFs into structured text."""
import pdfplumber

class PDFParser:
    def parse_pdf(self, pdf_path: str):
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue
                pages.append(
                    {
                        "page": page_number,
                        "text": text
                    }
                )
        return pages