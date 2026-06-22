from src.ingestion.pdf_parser import PDFParser
from src.ingestion.parsers.income_tax_parser import (
    IncomeTaxParser,
)

# ==========================================================
# LOAD PDF
# ==========================================================

pages = PDFParser().parse_pdf(
    "data/raw/income_tax_act_1961.pdf"
)

# ==========================================================
# PARSE
# ==========================================================

parser = IncomeTaxParser()

sections = parser.parse(
    pages
)

# ==========================================================
# STATS
# ==========================================================

print("=" * 100)
print(
    f"TOTAL SECTIONS: {len(sections)}"
)
print("=" * 100)

# ==========================================================
# FIRST 15
# ==========================================================

print("\nFIRST 15 SECTIONS\n")

for section in sections[:15]:

    print(
        f"Section: {section.section_number}"
    )

    print(
        f"Chapter: {section.chapter}"
    )

    print(
        f"Title: {section.title}"
    )

    print(
        f"Page: {section.page}"
    )

    print(
        f"Text Length: {len(section.text)}"
    )

    print(
        f"Footnotes: {len(section.footnotes or [])}"
    )

    print("-" * 80)

# ==========================================================
# LAST 15
# ==========================================================

print("\nLAST 15 SECTIONS\n")

for section in sections[-15:]:

    print(
        f"Section: {section.section_number}"
    )

    print(
        f"Chapter: {section.chapter}"
    )

    print(
        f"Title: {section.title}"
    )

    print(
        f"Page: {section.page}"
    )

    print(
        f"Text Length: {len(section.text)}"
    )

    print(
        f"Footnotes: {len(section.footnotes or [])}"
    )

    print("-" * 80)

# ==========================================================
# DEEP VALIDATION
# ==========================================================

TARGETS = [
    "1",
    "2",
    "10",
    "10A",
    "10AA",
    "80C",
    "115BAC",
    "115BBE",
    "139",
    "143",
    "147",
    "271",
]

for target in TARGETS:

    found = None

    for section in sections:

        if section.section_number == target:

            found = section
            break

    if not found:
        continue

    print("\n")
    print("=" * 100)

    print(
        f"SECTION {target}"
    )

    print("=" * 100)

    print(
        f"Chapter: {found.chapter}"
    )

    print(
        f"Title: {found.title}"
    )

    print(
        f"Page: {found.page}"
    )

    print(
        f"Length: {len(found.text)}"
    )

    print(
        f"Footnotes: {len(found.footnotes or [])}"
    )

    print("\nPREVIEW:\n")

    print(
        found.text[:2500]
    )

# ==========================================================
# LARGEST SECTIONS
# ==========================================================

print("\n")
print("=" * 100)
print("TOP 10 LARGEST SECTIONS")
print("=" * 100)

largest = sorted(
    sections,
    key=lambda x: len(x.text),
    reverse=True,
)

for section in largest[:10]:

    print(
        f"{section.section_number:<10}"
        f"{len(section.text):>10}"
    )

# ==========================================================
# SUMMARY
# ==========================================================

print("\n")
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    f"Total Sections Parsed: {len(sections)}"
)

avg_length = sum(
    len(s.text)
    for s in sections
) / len(sections)

print(
    f"Average Section Length: {avg_length:.2f}"
)

print(
    f"Longest Section Length: "
    f"{max(len(s.text) for s in sections)}"
)

print(
    f"Shortest Section Length: "
    f"{min(len(s.text) for s in sections)}"
)