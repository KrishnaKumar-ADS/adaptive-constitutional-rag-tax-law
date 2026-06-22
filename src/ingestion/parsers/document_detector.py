class DocumentDetector:
    @staticmethod
    def detect(
        first_pages_text: str
    ) -> str:

        text = first_pages_text.upper()

        if "INCOME-TAX ACT" in text:
            return "income_tax_act"

        if "CONSTITUTION OF INDIA" in text:
            return "constitution"

        raise ValueError(
            "Unknown document type"
        )