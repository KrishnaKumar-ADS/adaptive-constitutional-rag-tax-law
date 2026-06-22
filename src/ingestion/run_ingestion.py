import json
from pathlib import Path

from src.ingestion.pdf_parser import PDFParser

from src.ingestion.parsers.constitution_parser import (
    ConstitutionParser,
)

from src.ingestion.parsers.income_tax_parser import (
    IncomeTaxParser,
)

from src.ingestion.metadata_extractor import (
    MetadataExtractor,
)

from src.ingestion.chunker import (
    LegalChunker,
)


RAW_DIR = Path("data/raw")

PROCESSED_DIR = Path(
    "data/processed"
)

CONSTITUTION_PDF = (
    RAW_DIR
    / "constitution_of_india.pdf"
)

INCOME_TAX_PDF = (
    RAW_DIR
    / "income_tax_act_1961.pdf"
)

CHUNKS_FILE = (
    PROCESSED_DIR
    / "chunks.jsonl"
)

STATS_FILE = (
    PROCESSED_DIR
    / "ingestion_stats.json"
)


def write_chunks(
    chunks,
):

    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        for chunk in chunks:

            record = {
                "chunk_id":
                    chunk.chunk_id,

                "text":
                    chunk.text,

                "metadata":
                    chunk.metadata,
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )

            f.write("\n")


def write_stats(
    stats,
):

    with open(
        STATS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            stats,
            f,
            indent=4,
            ensure_ascii=False,
        )


def main():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_parser = PDFParser()

    metadata_extractor = (
        MetadataExtractor()
    )

    chunker = LegalChunker()

    # ======================================================
    # CONSTITUTION
    # ======================================================

    print(
        "\nParsing Constitution..."
    )

    constitution_pages = (
        pdf_parser.parse_pdf(
            str(
                CONSTITUTION_PDF
            )
        )
    )

    constitution_units = (
        ConstitutionParser()
        .parse(
            constitution_pages
        )
    )

    constitution_documents = [

        metadata_extractor.extract(
            unit
        )

        for unit in constitution_units
    ]

    constitution_chunks = (
        chunker.chunk_documents(
            constitution_documents
        )
    )

    # ======================================================
    # INCOME TAX
    # ======================================================

    print(
        "\nParsing Income Tax Act..."
    )

    income_tax_pages = (
        pdf_parser.parse_pdf(
            str(
                INCOME_TAX_PDF
            )
        )
    )

    income_tax_units = (
        IncomeTaxParser()
        .parse(
            income_tax_pages
        )
    )

    income_tax_documents = [
        metadata_extractor.extract(
            unit
        )
        for unit in income_tax_units
    ]
    income_tax_chunks = (
        chunker.chunk_documents(
            income_tax_documents
        )
    )

    # ======================================================
    # MERGE
    # ======================================================

    all_chunks = (
        constitution_chunks
        + income_tax_chunks
    )

    # ======================================================
    # WRITE FILES
    # ======================================================

    write_chunks(
        all_chunks
    )
    stats = {
        "constitution_articles":
            len(
                constitution_units
            ),
        "constitution_chunks":
            len(
                constitution_chunks
            ),
        "income_tax_sections":
            len(
                income_tax_units
            ),
        "income_tax_chunks":
            len(
                income_tax_chunks
            ),
        "total_chunks":
            len(
                all_chunks
            ),
    }
    write_stats(
        stats
    )

    # ======================================================
    # SUMMARY
    # ======================================================
    print(
        "\n"
        + "=" * 60
    )
    print(
        "INGESTION COMPLETE"
    )
    print(
        "=" * 60
    )
    print(
        f"Constitution Articles: "
        f"{len(constitution_units)}"
    )
    print(
        f"Constitution Chunks: "
        f"{len(constitution_chunks)}"
    )
    print(
        f"Income Tax Sections: "
        f"{len(income_tax_units)}"
    )
    print(
        f"Income Tax Chunks: "
        f"{len(income_tax_chunks)}"
    )
    print(
        f"Total Chunks: "
        f"{len(all_chunks)}"
    )
    print(
        f"\nSaved:"
    )
    print(
        CHUNKS_FILE
    )
    print(
        STATS_FILE
    )

if __name__ == "__main__":
    main()