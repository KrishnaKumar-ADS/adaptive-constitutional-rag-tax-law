"""Check whether a citation was actually grounded in retrieved evidence."""
import re


class GroundingChecker:

    @staticmethod
    def normalize(value: str) -> str:
        value = str(value).lower().strip()

        # 10(1) -> 10
        value = re.sub(r"\(\d+\)", "", value)

        value = value.replace("section", "")
        value = value.replace("article", "")

        return value.strip()

    @classmethod
    def citation_used(
        cls,
        citation: str,
        evidence_list,
    ) -> bool:
        """
        Check whether a cited section/article was actually present
        in the retrieved evidence.

        evidence_list can be:
          - list of Evidence dataclass objects (from evidence_aggregator)
          - list of dicts with keys like section_number, article_number, citation_id
        """
        citation_norm = cls.normalize(citation)

        for evidence in evidence_list:

            # Handle both dict and dataclass Evidence objects
            if hasattr(evidence, "section_number"):
                section = cls.normalize(
                    evidence.section_number or ""
                )
                article = cls.normalize(
                    evidence.article_number or ""
                )
                cit_id = cls.normalize(
                    evidence.citation_id or ""
                )
            else:
                section = cls.normalize(
                    evidence.get("section_number", "")
                )
                article = cls.normalize(
                    evidence.get("article_number", "")
                )
                cit_id = cls.normalize(
                    evidence.get("citation_id", "")
                )

            if citation_norm and citation_norm == section:
                return True

            if citation_norm and citation_norm == article:
                return True

            if citation_norm and citation_norm in cit_id:
                return True

            # ALSO check if it's in the raw text, as metadata might be incomplete
            if hasattr(evidence, "text"):
                text = getattr(evidence, "text") or ""
            else:
                text = evidence.get("text", "")
            
            # Simple substring check (e.g. '206c' in lowercased text)
            if citation_norm and citation_norm in str(text).lower():
                return True

            # Also check the raw citation string just in case
            if str(citation).lower() in str(text).lower():
                return True

        return False