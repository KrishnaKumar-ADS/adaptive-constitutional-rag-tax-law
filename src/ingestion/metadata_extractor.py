class MetadataExtractor:

    def extract(
        self,
        legal_unit,
    ):

        metadata = {
            "source": legal_unit.source,
            "page": legal_unit.page,
            "title": legal_unit.title,
        }

        # =====================================
        # CONSTITUTION
        # =====================================

        if (
            legal_unit.source
            == "constitution"
        ):

            unit_id = (
                f"constitution_article_"
                f"{legal_unit.article_number}"
            )

            metadata.update(
                {
                    "document_type": "constitution",
                    "part": legal_unit.part,
                    "article_number":
                        legal_unit.article_number,
                    "citation":
                        f"Article {legal_unit.article_number}, Constitution of India",
                }
            )

        # =====================================
        # INCOME TAX
        # =====================================

        else:

            unit_id = (
                f"income_tax_section_"
                f"{legal_unit.section_number}"
            )

            metadata.update(
                {
                    "document_type": "statute",
                    "chapter":
                        legal_unit.chapter,
                    "section_number":
                        legal_unit.section_number,
                    "citation":
                        f"Section {legal_unit.section_number}, Income-tax Act, 1961",
                }
            )

        metadata["unit_id"] = unit_id

        return {
            "unit_id": unit_id,
            "metadata": metadata,
            "text": legal_unit.text,
        }