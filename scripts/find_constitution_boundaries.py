from src.ingestion.parsers.constitution_parser import (
    parse_constitution
)

PDF_PATH = "data/raw/constitution_of_india.pdf"


def main():

    articles = parse_constitution(
        PDF_PATH
    )

    print("=" * 80)
    print(f"TOTAL ARTICLES: {len(articles)}")
    print("=" * 80)

    print("\nFIRST 10 ARTICLES\n")

    for article in articles[:10]:

        print(
            f"Article: {article['article_number']}"
        )

        print(
            f"Part: {article['part']}"
        )

        print(
            f"Title: {article['title']}"
        )

        print(
            f"Content Length: {len(article['content'])}"
        )

        print("-" * 80)

    print("\nLAST 10 ARTICLES\n")

    for article in articles[-10:]:

        print(
            f"Article: {article['article_number']}"
        )

        print(
            f"Part: {article['part']}"
        )

        print(
            f"Title: {article['title']}"
        )

        print(
            f"Content Length: {len(article['content'])}"
        )

        print("-" * 80)


if __name__ == "__main__":
    main()