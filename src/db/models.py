"""SQLAlchemy database models."""
# src/db/models.py
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import DateTime
from sqlalchemy import Text
from sqlalchemy import Float
from sqlalchemy.sql import func
from src.db.database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(
        Integer,
        primary_key=True,
        index=True
    )
    username = Column(
        String,
        unique=True,
        nullable=False
    )
    email = Column(
        String,
        unique=True,
        nullable=False
    )
    hashed_password = Column(
        String,
        nullable=False
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

class QueryLog(Base):
    __tablename__ = "query_logs"
    id = Column(
        Integer,
        primary_key=True,
        index=True
    )
    question = Column(
        Text,
        nullable=False
    )
    uncertainty_score = Column(
        Float,
        nullable=True
    )
    decision = Column(
        String,
        nullable=True
    )
    answer = Column(
        Text,
        nullable=True
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )