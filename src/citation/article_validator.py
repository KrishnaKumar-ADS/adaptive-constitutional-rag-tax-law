"""Validate constitutional article references."""
import json
import re
from pathlib import Path


class ArticleValidator:
    ARTICLE_PATTERN = re.compile(
        r"Article\s+(\d+[A-Z]*)",
        re.IGNORECASE
    )

    def __init__(
        self,
        index_path: str = "data/processed/article_index.json"
    ):
        self.index = json.loads(
            Path(index_path).read_text(
                encoding="utf-8"
            )
        )

    def extract_articles(self, text: str):
        return list(
            set(
                self.ARTICLE_PATTERN.findall(text)
            )
        )

    def validate(self, text: str):

        articles = self.extract_articles(text)

        results = []

        for article in articles:

            exists = (
                article in self.index
                or f"Article {article}" in self.index
            )

            results.append(
                {
                    "article": article,
                    "exists": exists
                }
            )

        return results