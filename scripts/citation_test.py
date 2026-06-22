import json

sections = set()

with open(
    "data/processed/chunks.jsonl",
    encoding="utf-8"
) as f:

    for line in f:

        chunk = json.loads(line)

        sec = chunk["metadata"].get(
            "section_number"
        )

        if sec:
            sections.add(str(sec))

print("80C" in sections)
print("115BAC" in sections)