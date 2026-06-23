"""Validate income tax section references against ground-truth index."""
import json
import re
from pathlib import Path


class SectionValidator:
    SECTION_PATTERN = re.compile(
        r"Section\s+(\d+[A-Z]*(?:\(\d+\))?)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        index_path: str = "data/processed/section_index.json",
    ):
        self.index = json.loads(
            Path(index_path).read_text(
                encoding="utf-8"
            )
        )

    def extract_sections(self, text: str) -> list[str]:
        return list(
            set(
                self.SECTION_PATTERN.findall(text)
            )
        )

    def validate(self, text: str):
        sections = self.extract_sections(text)

        results = []

        for section in sections:

            # Section 10(1) -> base = 10
            base_section = re.sub(
                r"\(\d+\)",
                "",
                section,
            )

            exists = (
                f"Section {section}" in self.index
                or f"Section {base_section}" in self.index
            )

            results.append(
                {
                    "section": section,
                    "exists": exists,
                }
            )

        return results