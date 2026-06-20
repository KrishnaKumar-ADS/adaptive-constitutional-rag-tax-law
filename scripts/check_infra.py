"""Verify infrastructure services."""
import psycopg2
import redis
from qdrant_client import QdrantClient
from src.config import settings

def test_postgres():
    conn = psycopg2.connect(settings.DATABASE_URL)
    conn.close()
    print("PostgreSQL OK")


def test_redis():
    r = redis.from_url(settings.REDIS_URL)
    r.ping()
    print("Redis OK")

def test_qdrant():
    client = QdrantClient(url=settings.QDRANT_URL)
    client.get_collections()
    print("Qdrant OK")

if __name__ == "__main__":
    test_postgres()
    test_redis()
    test_qdrant()
    print("\nAll Infrastructure Running")