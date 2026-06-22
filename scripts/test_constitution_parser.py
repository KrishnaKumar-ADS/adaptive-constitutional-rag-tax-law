from src.ingestion.pdf_parser import PDFParser

from src.ingestion.parsers.constitution_parser import (
    ConstitutionParser,
)

# --------------------------------------------------
# Load PDF
# --------------------------------------------------

pages = PDFParser().parse_pdf(
    "data/raw/constitution_of_india.pdf"
)

# --------------------------------------------------
# Parse
# --------------------------------------------------

parser = ConstitutionParser()

articles = parser.parse(
    pages
)

# --------------------------------------------------
# Stats
# --------------------------------------------------

print("=" * 100)
print(
    f"TOTAL ARTICLES: {len(articles)}"
)
print("=" * 100)

# --------------------------------------------------
# FIRST 15
# --------------------------------------------------

print("\nFIRST 15 ARTICLES\n")

for article in articles[:15]:

    print(
        f"Article: {article.article_number}"
    )

    print(
        f"Part: {article.part}"
    )

    print(
        f"Title: {article.title}"
    )

    print(
        f"Page: {article.page}"
    )

    print(
        f"Text Length: {len(article.text)}"
    )

    print(
        f"Footnotes: {len(article.footnotes or [])}"
    )

    print("-" * 80)

# --------------------------------------------------
# LAST 15
# --------------------------------------------------

print("\nLAST 15 ARTICLES\n")

for article in articles[-15:]:

    print(
        f"Article: {article.article_number}"
    )

    print(
        f"Part: {article.part}"
    )

    print(
        f"Title: {article.title}"
    )

    print(
        f"Page: {article.page}"
    )

    print(
        f"Text Length: {len(article.text)}"
    )

    print(
        f"Footnotes: {len(article.footnotes or [])}"
    )

    print("-" * 80)

# --------------------------------------------------
# ARTICLE 1 FULL PREVIEW
# --------------------------------------------------

print("\n")
print("=" * 100)
print("ARTICLE 1 PREVIEW")
print("=" * 100)

article_1 = articles[0]

print(
    article_1.text[:3000]
)

# --------------------------------------------------
# ARTICLE 21 PREVIEW
# --------------------------------------------------

article_21 = None

for article in articles:

    if article.article_number == "21":

        article_21 = article

        break

if article_21:

    print("\n")
    print("=" * 100)
    print("ARTICLE 21 PREVIEW")
    print("=" * 100)

    print(
        article_21.text[:3000]
    )

# --------------------------------------------------
# ARTICLE 372 FOOTNOTE TEST
# --------------------------------------------------

article_372 = None

for article in articles:

    if article.article_number == "372":

        article_372 = article

        break

if article_372:

    print("\n")
    print("=" * 100)
    print("ARTICLE 372")
    print("=" * 100)

    print(
        f"Footnotes: {len(article_372.footnotes or [])}"
    )

    if article_372.footnotes:

        print("\nFIRST FOOTNOTE:\n")

        print(
            article_372.footnotes[0][:1500]
        )