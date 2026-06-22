from src.ingestion.pdf_parser import PDFParser

from src.ingestion.parsers.constitution_parser import (
    ConstitutionParser,
)

from src.ingestion.metadata_extractor import (
    MetadataExtractor,
)

from src.ingestion.chunker import (
    LegalChunker,
)

pages = PDFParser().parse_pdf(
    "data/raw/constitution_of_india.pdf"
)

units = ConstitutionParser().parse(
    pages
)

extractor = MetadataExtractor()

documents = [
    extractor.extract(unit)
    for unit in units
]

chunker = LegalChunker()

chunks = chunker.chunk_documents(
    documents
)

print(
    f"Units: {len(units)}"
)

print(
    f"Documents: {len(documents)}"
)

print(
    f"Chunks: {len(chunks)}"
)

print()

for chunk in chunks[:10]:

    print(
        chunk.chunk_id
    )

    print(
        chunk.metadata
    )

    print(
        f"Length: {len(chunk.text)}"
    )

    print("-" * 80)

print()
print("=" * 100)
print("LARGEST CHUNK")
print("=" * 100)

largest = max(
    chunks,
    key=lambda x: len(x.text)
)

print(
    largest.chunk_id
)

print(
    len(
        largest.text
    )
)