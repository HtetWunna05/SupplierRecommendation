import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
uri = os.getenv("MONGODB_URI")

def connect_to_mongodb():
    try:
        client = MongoClient(uri)
        # Test connection
        client.admin.command('ping')
        db = client['supplier_db']
        return db
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None

# This is the function your app.py was missing!
def create_database():
    db = connect_to_mongodb()
    if db is not None:
        print("✅ Database link is active and ready!")
    return db

if __name__ == "__main__":
    connect_to_mongodb()