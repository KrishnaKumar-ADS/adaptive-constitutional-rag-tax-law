from src.retrieval.hybrid_retriever import hybrid_search

TEST_CASES = [
    {
        "query": "Is agricultural income taxable?",
        "expected_section": "10(1)",
        "expected_article": None,
    },
    {
        "query": "What does Article 265 say about taxation?",
        "expected_section": None,
        "expected_article": "265",
    },
    {
        "query": "What is Permanent Account Number?",
        "expected_section": "139A",
        "expected_article": None,
    },
    {
        "query": "Income escaping assessment",
        "expected_section": "147",
        "expected_article": None,
    },
    {
        "query": "What is advance tax?",
        "expected_section": "208",
        "expected_article": None,
    },
    {
        "query": "Tax deducted at source salary",
        "expected_section": "192",
        "expected_article": None,
    },
    {
        "query": "Deduction for higher education loan",
        "expected_section": "80E",
        "expected_article": None,
    },
    {
        "query": "Health insurance deduction",
        "expected_section": "80D",
        "expected_article": None,
    },
    {
        "query": "Right to equality",
        "expected_section": None,
        "expected_article": "14",
    },
    {
        "query": "Freedom of speech",
        "expected_section": None,
        "expected_article": "19",
    },
    {
        "query": "What is best judgment assessment?",
        "expected_section": "144",
        "expected_article": None,
    },
    {
        "query": "Who is liable to pay advance tax?",
        "expected_section": "208",
        "expected_article": None,
    },
    {
        "query": "Can tax be imposed without authority of law?",
        "expected_section": None,
        "expected_article": "265",
    },
    {
        "query": "What is reassessment?",
        "expected_section": "147",
        "expected_article": None,
    },
    {
        "query": "What deduction is available for medical insurance?",
        "expected_section": "80D",
        "expected_article": None,
    },
]

passed = 0
total = len(TEST_CASES)

print("\n" + "=" * 80)
print("RETRIEVAL EVALUATION")
print("=" * 80)

for idx, test in enumerate(TEST_CASES, start=1):

    query = test["query"]
    expected_section = test["expected_section"]
    expected_article = test["expected_article"]

    print(f"\n[{idx}/{total}] {query}")

    try:
        results = hybrid_search(query)

        found = False
        found_rank = None

        for rank, result in enumerate(results, start=1):

            payload = result.payload

            section = str(payload.get("section_number", "")).strip()
            article = str(payload.get("article_number", "")).strip()

            if expected_section and section == expected_section:
                found = True
                found_rank = rank
                break

            if expected_article and article == expected_article:
                found = True
                found_rank = rank
                break

        if found:
            passed += 1
            print(f"✅ PASS (Rank {found_rank})")
        else:
            print("❌ FAIL")

            print("\nTop Results:")

            for rank, result in enumerate(results[:5], start=1):

                payload = result.payload

                print(
                    f"{rank}. "
                    f"Section={payload.get('section_number')} "
                    f"Article={payload.get('article_number')}"
                )

    except Exception as e:
        print(f"❌ ERROR: {e}")

accuracy = (passed / total) * 100

print("\n" + "=" * 80)
print(f"Passed : {passed}/{total}")
print(f"Accuracy : {accuracy:.2f}%")
print("=" * 80)



results = hybrid_search("What is reassessment?", top_k=20)

for i, r in enumerate(results, 1):
    print(
        i,
        r.payload.get("section_number"),
        r.score
    )