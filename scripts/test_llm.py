import asyncio

from src.evidence.evidence_aggregator import (
    Evidence,
    EvidenceSet,
)

from src.llm.prompts import (
    build_prompt,
)

from src.llm.inference_openrouter import (
    generate_answer,
)


evidence = Evidence(
    citation_id="Section 10(1)",
    text="""
Agricultural income shall not be
included in total income.
""",
    source_type="income_tax_act",
    score=1.0,
)

evset = EvidenceSet(
    evidences=[evidence]
)


prompt = build_prompt(
    "Is agricultural income taxable?",
    evset,
)


async def main():

    answer = await generate_answer(
        question="Is agricultural income taxable?",
        prompt=prompt,
    )

    print(answer)


asyncio.run(main())