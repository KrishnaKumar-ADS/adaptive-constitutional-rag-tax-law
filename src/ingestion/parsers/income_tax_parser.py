import re

from src.ingestion.parsers.base_parser import (
    BaseParser,
    LegalUnit,
)

# ==========================================================
# PATTERNS
# ==========================================================

CHAPTER_PATTERN = re.compile(
    r"(?m)^CHAPTER\s+([IVXLC]+(?:-[A-Z]+)?)\s*$"
)

SECTION_PATTERN = re.compile(
    r"(?m)^(?:\d+\[)?(\d+[A-Z]{0,5})\.\s"
)

FOOTNOTE_SEPARATOR = re.compile(
    r"_{10,}"
)

INVALID_TITLE_PREFIXES = (
    "Subs",
    "Ins",
    "Inserted",
    "Substituted",
    "Omitted",
    "Clause",
    "Explanation",
    "The words",
    "For ",
    "In ",
    "After ",
    "Before ",
)

# ==========================================================
# FOOTNOTE EXTRACTION
# ==========================================================

def split_main_text_and_footnotes(
    section_text: str,
):

    parts = FOOTNOTE_SEPARATOR.split(
        section_text,
        maxsplit=1,
    )

    if len(parts) == 1:

        return (
            section_text.strip(),
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

            current.append(line)

    if current:

        footnotes.append(
            " ".join(current)
        )

    return (
        main_text,
        footnotes,
    )


# ==========================================================
# INCOME TAX PARSER
# ==========================================================

class IncomeTaxParser(BaseParser):

    START_PAGE = 30

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

        # ==========================================
        # CHAPTER DETECTION
        # ==========================================

        chapter_positions = []

        for match in CHAPTER_PATTERN.finditer(
            document_text
        ):

            chapter_positions.append(
                (
                    match.start(),
                    f"CHAPTER {match.group(1)}",
                )
            )

        # ==========================================
        # SECTION DETECTION
        # ==========================================

        section_matches = list(
            SECTION_PATTERN.finditer(
                document_text
            )
        )

        legal_units = []

        for i, match in enumerate(
            section_matches
        ):

            section_number = (
                match.group(1)
            )

            start = match.start()

            end = (
                section_matches[i + 1].start()
                if i + 1 < len(section_matches)
                else len(document_text)
            )

            section_text = (
                document_text[start:end]
                .strip()
            )

            if not section_text:
                continue

            # ======================================
            # CHAPTER
            # ======================================

            current_chapter = None

            for (
                pos,
                chapter,
            ) in chapter_positions:

                if pos <= start:

                    current_chapter = (
                        chapter
                    )

                else:

                    break

            # ======================================
            # PAGE
            # ======================================

            current_page = (
                self.START_PAGE
            )

            for (
                pos,
                page_no,
            ) in page_map:

                if pos <= start:

                    current_page = (
                        page_no
                    )

                else:

                    break

            # ======================================
            # TITLE
            # ======================================

            # Robust title match: look at the first 300 characters of section text (replacing newlines with spaces)
            # to handle long/wrapped titles like Section 206C.
            title_text = section_text[:300].replace("\n", " ").strip()
            title = None
            title_match = re.match(
                r"(?:\d+\[)?\d+[A-Z]{0,5}\.\s*(.*?)(?:—|\.)",
                title_text,
            )

            if title_match:
                title = (
                    title_match.group(1)
                    .strip()
                )

            if not title:
                continue

            # ======================================
            # FOOTNOTES
            # ======================================

            (
                main_text,
                footnotes,
            ) = split_main_text_and_footnotes(
                section_text
            )

            # ======================================
            # FILTER AMENDMENT NOTES
            # ======================================

            if title.startswith(
                INVALID_TITLE_PREFIXES
            ):
                continue

            if (
                "Act" in title
                and "w.e.f."
                in section_text[:500]
            ):
                continue

            lower_title = (
                title.lower()
            )

            if (
                "renumbered"
                in lower_title
            ):
                continue

            if (
                "omitted"
                in lower_title
            ):
                continue

            if (
                "inserted"
                in lower_title
            ):
                continue

            if (
                "substituted"
                in lower_title
            ):
                continue

            if (
                len(main_text) < 100
            ):
                continue

            # ======================================
            # CREATE LEGAL UNIT
            # ======================================

            legal_units.append(
                LegalUnit(
                    source="income_tax_act",
                    chapter=current_chapter,
                    section_number=section_number,
                    title=title,
                    page=current_page,
                    text=main_text,
                    footnotes=footnotes,
                )
            )

        return legal_units