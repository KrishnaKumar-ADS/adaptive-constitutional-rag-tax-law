"""SQLAlchemy database models."""
# src/db/models.py
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import DateTime
from sqlalchemy import Text
from sqlalchemy import Float
from sqlalchemy import JSON
from sqlalchemy.sql import func
from src.db.database import Base


class QueryLog(Base):
    """
    Matches the guide2.md Day 1 spec:
    id, question, evidence_ids, uncertainty_score,
    decision, citations_json, raw_response, created_at
    """
    __tablename__ = "queries"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )
    question = Column(
        Text,
        nullable=False,
    )
    evidence_ids = Column(
        JSON,
        nullable=True,
    )
    uncertainty_score = Column(
        Float,
        nullable=True,
    )
    decision = Column(
        String,
        nullable=True,
    )
    citations_json = Column(
        JSON,
        nullable=True,
    )
    raw_response = Column(
        Text,
        nullable=True,
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )