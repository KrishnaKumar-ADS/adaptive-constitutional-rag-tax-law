from dataclasses import dataclass

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


@dataclass
class Chunk:

    chunk_id: str

    text: str

    metadata: dict


class LegalChunker:

    def __init__(
        self,
        chunk_size=1200,
        chunk_overlap=200,
    ):

        self.splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    "; ",
                    " ",
                ],
            )
        )

    def build_hierarchy(
        self,
        metadata,
    ):

        source = metadata.get(
            "source"
        )

        if source == "constitution":

            return (
                f"{metadata.get('part')} > "
                f"Article {metadata.get('article_number')}"
            )

        if source == "income_tax_act":

            return (
                f"{metadata.get('chapter')} > "
                f"Section {metadata.get('section_number')}"
            )

        return ""

    def chunk_unit(
        self,
        unit_document,
    ):

        chunks = []

        split_texts = (
            self.splitter.split_text(
                unit_document["text"]
            )
        )

        unit_id = (
            unit_document["unit_id"]
        )

        metadata = (
            unit_document["metadata"]
        )

        total_chunks = len(
            split_texts
        )

        for idx, text in enumerate(
            split_texts,
            start=1,
        ):

            chunk_id = (
                f"{unit_id}_chunk_{idx}"
            )

            chunk_metadata = {
                **metadata,

                "parent_unit_id":
                    unit_id,

                "chunk_id":
                    chunk_id,

                "chunk_index":
                    idx,

                "total_chunks":
                    total_chunks,

                "legal_hierarchy":
                    self.build_hierarchy(
                        metadata
                    ),
            }

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=chunk_metadata,
                )
            )

        return chunks

    def chunk_documents(
        self,
        documents,
    ):

        all_chunks = []

        for document in documents:

            all_chunks.extend(
                self.chunk_unit(
                    document
                )
            )

        return all_chunks