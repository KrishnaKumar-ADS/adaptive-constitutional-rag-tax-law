# generate_indexes.py

import json
import re
from collections import defaultdict
from pathlib import Path


INPUT_FILE = "data/processed/chunks.jsonl"

SECTION_OUTPUT = "data/processed/section_index.json"
ARTICLE_OUTPUT = "data/processed/article_index.json"


# --------------------------------------------------
# Subsection extraction
# --------------------------------------------------

SUBSECTION_PATTERN = re.compile(
    r'\(([0-9A-Za-z]+)\)'
)


def extract_subsections(text: str):
    """
    Finds:
        (1)
        (2)
        (2A)
        (A)
        etc.

    Returns:
        {
            "1": "...",
            "2": "...",
            "2A": "..."
        }
    """

    matches = list(SUBSECTION_PATTERN.finditer(text))

    if not matches:
        return {}

    subsections = {}

    for i, match in enumerate(matches):
        subsection_id = match.group(1)

        start = match.start()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        subsection_text = text[start:end].strip()

        if subsection_id not in subsections:
            subsections[subsection_id] = subsection_text
        else:
            subsections[subsection_id] += "\n" + subsection_text

    return subsections


# --------------------------------------------------
# Main
# --------------------------------------------------

section_index = defaultdict(
    lambda: {
        "section_number": None,
        "title": "",
        "citation": "",
        "page": None,
        "chunk_ids": [],
        "text": "",
        "subsections": {}
    }
)

article_index = defaultdict(
    lambda: {
        "article_number": None,
        "title": "",
        "citation": "",
        "page": None,
        "chunk_ids": [],
        "text": ""
    }
)


with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:
        chunk = json.loads(line)

        chunk_id = chunk["chunk_id"]
        text = chunk["text"]

        md = chunk["metadata"]

        # --------------------------------------------------
        # INCOME TAX SECTIONS
        # --------------------------------------------------

        if "section_number" in md:

            sec_no = str(md["section_number"]).strip()

            key = f"Section {sec_no}"

            entry = section_index[key]

            entry["section_number"] = sec_no
            entry["title"] = md.get("title", "")
            entry["citation"] = md.get(
                "citation",
                f"Section {sec_no}"
            )

            if entry["page"] is None:
                entry["page"] = md.get("page")

            entry["chunk_ids"].append(chunk_id)

            if entry["text"]:
                entry["text"] += "\n\n" + text
            else:
                entry["text"] = text

            subs = extract_subsections(text)

            for sub_id, sub_text in subs.items():

                if sub_id not in entry["subsections"]:
                    entry["subsections"][sub_id] = {
                        "text": sub_text,
                        "chunk_ids": [chunk_id]
                    }

                else:
                    entry["subsections"][sub_id]["text"] += (
                        "\n" + sub_text
                    )

                    entry["subsections"][sub_id][
                        "chunk_ids"
                    ].append(chunk_id)

        # --------------------------------------------------
        # CONSTITUTION ARTICLES
        # --------------------------------------------------

        elif "article_number" in md:

            art_no = str(md["article_number"]).strip()

            key = f"Article {art_no}"

            entry = article_index[key]

            entry["article_number"] = art_no
            entry["title"] = md.get("title", "")
            entry["citation"] = md.get(
                "citation",
                f"Article {art_no}"
            )

            if entry["page"] is None:
                entry["page"] = md.get("page")

            entry["chunk_ids"].append(chunk_id)

            if entry["text"]:
                entry["text"] += "\n\n" + text
            else:
                entry["text"] = text


# --------------------------------------------------
# Save
# --------------------------------------------------

Path(SECTION_OUTPUT).parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    SECTION_OUTPUT,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        dict(section_index),
        f,
        ensure_ascii=False,
        indent=2
    )

with open(
    ARTICLE_OUTPUT,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        dict(article_index),
        f,
        ensure_ascii=False,
        indent=2
    )

print(
    f"Saved {len(section_index)} sections "
    f"to {SECTION_OUTPUT}"
)

print(
    f"Saved {len(article_index)} articles "
    f"to {ARTICLE_OUTPUT}"
)