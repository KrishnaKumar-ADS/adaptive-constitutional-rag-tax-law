from src.db.database import SessionLocal
from src.db.models import User

db = SessionLocal()

user = User(
    username="krishna",
    email="krishna@test.com",
    hashed_password="dummyhash"
)
db.add(user)
db.commit()
print("User Added")
db.close()