import re

from src.ingestion.parsers.base_parser import (
    BaseParser,
    LegalUnit,
)

# ==========================================================
# PATTERNS
# ==========================================================

PART_PATTERN = re.compile(
    r"(?m)^PART\s+([IVXLC]+(?:-[A-Z]+)?)\s*$"
)

ARTICLE_PATTERN = re.compile(
    r"(?m)^(?:\d+\[)?(\d+[A-Z]{0,3})\.\s"
)

FOOTNOTE_SEPARATOR = re.compile(
    r"_{10,}"
)

INVALID_TITLE_PREFIXES = (
    "Subs. by",
    "Ins. by",
    "Inserted by",
    "Substituted by",
    "Omitted by",
    "The words",
    "Art.",
    "See ",
)

# ==========================================================
# FOOTNOTE EXTRACTION
# ==========================================================


def split_main_text_and_footnotes(
    article_text: str,
):

    parts = FOOTNOTE_SEPARATOR.split(
        article_text,
        maxsplit=1,
    )

    if len(parts) == 1:

        return (
            article_text.strip(),
            [],
        )

    main_text = parts[0].strip()

    footnote_block = parts[1]

    footnotes = []

    current = []

    for line in footnote_block.splitlines():

        line = line.strip()

        if not line:
            continue

        if re.match(
            r"^(\*|\d+\.)",
            line,
        ):

            if current:

                footnotes.append(
                    " ".join(current)
                )

            current = [line]

        else:

            current.append(
                line
            )

    if current:

        footnotes.append(
            " ".join(current)
        )

    return (
        main_text,
        footnotes,
    )


# ==========================================================
# CONSTITUTION PARSER
# ==========================================================


class ConstitutionParser(BaseParser):

    START_PAGE = 33
    END_PAGE = 284

    def parse(
        self,
        pages,
    ):

        document_parts = []

        page_map = []

        current_position = 0

        for page in pages:

            page_number = page["page"]

            if page_number < self.START_PAGE:
                continue

            if page_number > self.END_PAGE:
                break

            text = page["text"]

            page_map.append(
                (
                    current_position,
                    page_number,
                )
            )

            document_parts.append(
                text
            )

            current_position += (
                len(text) + 1
            )

        document_text = "\n".join(
            document_parts
        )

        # --------------------------------------
        # PART DETECTION
        # --------------------------------------

        part_positions = []

        for match in PART_PATTERN.finditer(
            document_text
        ):

            part_positions.append(
                (
                    match.start(),
                    f"PART {match.group(1)}",
                )
            )

        # --------------------------------------
        # ARTICLE DETECTION
        # --------------------------------------

        article_matches = list(
            ARTICLE_PATTERN.finditer(
                document_text
            )
        )

        legal_units = []

        for i, match in enumerate(
            article_matches
        ):

            article_number = (
                match.group(1)
            )

            start = match.start()

            end = (
                article_matches[i + 1].start()
                if i + 1 < len(article_matches)
                else len(document_text)
            )

            article_text = (
                document_text[start:end]
                .strip()
            )

            if not article_text:
                continue

            # --------------------------------------
            # SKIP FOOTNOTE REFERENCES
            # --------------------------------------

            if (
                "Constitution Amendment"
                in article_text[:200]
            ):
                continue

            # --------------------------------------
            # OMITTED ARTICLES
            # --------------------------------------

            if (
                "Omitted by the Constitution"
                in article_text[:500]
            ):
                continue

            # --------------------------------------
            # PART
            # --------------------------------------

            current_part = None

            for (
                part_pos,
                part_name,
            ) in part_positions:

                if part_pos <= start:
                    current_part = part_name
                else:
                    break

            # --------------------------------------
            # PAGE
            # --------------------------------------

            current_page = self.START_PAGE

            for (
                pos,
                page_no,
            ) in page_map:

                if pos <= start:
                    current_page = page_no
                else:
                    break

            # --------------------------------------
            # TITLE EXTRACTION
            # --------------------------------------

            first_line = (
                article_text
                .splitlines()[0]
                .strip()
            )

            title = re.sub(
                r"^(?:\d+\[)?\d+[A-Z]{0,3}\.\s*",
                "",
                first_line,
            )

            if "—" in title:
                title = title.split("—")[0]

            title = re.sub(
                r"^\d+\[",
                "",
                title,
            )

            title = title.replace(
                "[",
                ""
            ).replace(
                "]",
                ""
            )

            title = title.replace(
                "].",
                ""
            )

            title = (
                title.strip()
                .strip(".")
            )

            # --------------------------------------
            # FILTER FOOTNOTE ENTRIES
            # --------------------------------------

            if title.startswith(
                INVALID_TITLE_PREFIXES
            ):
                continue

            if "w.e.f." in title:
                continue

            if len(title) < 5:
                continue

            # --------------------------------------
            # FOOTNOTES
            # --------------------------------------

            (
                main_text,
                footnotes,
            ) = split_main_text_and_footnotes(
                article_text
            )

            legal_units.append(
                LegalUnit(
                    source="constitution",
                    part=current_part,
                    article_number=article_number,
                    title=title,
                    page=current_page,
                    text=main_text,
                    footnotes=footnotes,
                )
            )

        return legal_units