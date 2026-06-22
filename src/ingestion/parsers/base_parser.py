from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class LegalUnit:
    source: str
    text: str
    page: int
    chapter: Optional[str] = None
    part: Optional[str] = None
    section_number: Optional[str] = None
    article_number: Optional[str] = None
    subsection: Optional[str] = None
    title: Optional[str] = None
    footnotes: list[str] | None = None

class BaseParser(ABC):
    @abstractmethod
    def parse(
        self,
        pages
    ):
        pass