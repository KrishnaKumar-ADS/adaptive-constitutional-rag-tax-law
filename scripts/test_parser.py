from src.ingestion.pdf_parser import PDFParser
parser = PDFParser()
pages = parser.parse_pdf(
    "data/raw/constitution_of_india.pdf"
)
print("Total Pages:", len(pages))
print()
print(pages[0]["text"][:1000])