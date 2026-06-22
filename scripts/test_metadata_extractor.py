from src.ingestion.pdf_parser import PDFParser

from src.ingestion.parsers.constitution_parser import (
    ConstitutionParser,
)

from src.ingestion.metadata_extractor import (
    MetadataExtractor,
)

pages = PDFParser().parse_pdf(
    "data/raw/constitution_of_india.pdf"
)

units = ConstitutionParser().parse(
    pages
)

extractor = MetadataExtractor()

record = extractor.extract(
    units[0]
)

print(record)