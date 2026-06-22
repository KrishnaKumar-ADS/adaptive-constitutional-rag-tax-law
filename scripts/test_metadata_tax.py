from src.ingestion.pdf_parser import PDFParser

from src.ingestion.parsers.income_tax_parser import (
    IncomeTaxParser,
)

from src.ingestion.metadata_extractor import (
    MetadataExtractor,
)

pages = PDFParser().parse_pdf(
    "data/raw/income_tax_act_1961.pdf"
)

units = IncomeTaxParser().parse(
    pages
)

extractor = MetadataExtractor()

record = extractor.extract(
    units[0]
)

print(record)