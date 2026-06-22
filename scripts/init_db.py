from src.db.database import engine
from src.db.models import Base

print("Creating Tables...")
Base.metadata.create_all(bind=engine)
print("Tables Created")