import re


class LegalUnitValidator:

    SUSPICIOUS_WORDS = {
        "schedule",
        "appendix",
        "classification",
        "industrial",
        "item no",
        "table",
        "annexure",
    }

    def is_valid_section(self, record):

        title = (
            record.get("title", "")
            .lower()
            .strip()
        )

        section_number = str(
            record.get("section_number", "")
        )

        for word in self.SUSPICIOUS_WORDS:

            if word in title:
                return False

        if not re.match(
            r"^\d+[A-Z]*$",
            section_number
        ):
            return False

        try:

            numeric_part = int(
                re.match(
                    r"\d+",
                    section_number
                ).group()
            )

            if numeric_part > 300:
                return False

        except Exception:

            return False

        return True

    def is_valid_article(self, record):

        article_number = str(
            record.get(
                "article_number",
                ""
            )
        )

        if not re.match(
            r"^\d+[A-Z]*$",
            article_number
        ):
            return False

        return True