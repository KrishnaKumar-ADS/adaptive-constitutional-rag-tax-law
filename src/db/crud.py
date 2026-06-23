"""Database CRUD operations for query logging."""
import json
from sqlalchemy.orm import Session
from src.db.database import SessionLocal
from src.db.models import QueryLog


def log_query(
    question: str,
    evidence_ids: list = None,
    uncertainty_score: float = None,
    decision: str = None,
    citations_json: dict = None,
    raw_response: str = None,
):
    """Insert a query result into the queries table."""
    db: Session = SessionLocal()

    try:
        entry = QueryLog(
            question=question,
            evidence_ids=evidence_ids,
            uncertainty_score=uncertainty_score,
            decision=decision,
            citations_json=citations_json,
            raw_response=raw_response,
        )

        db.add(entry)
        db.commit()
        db.refresh(entry)

        return entry.id

    except Exception as e:
        db.rollback()
        print(f"DB logging failed: {e}")
        return None

    finally:
        db.close()
