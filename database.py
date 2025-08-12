# database.py
from __future__ import annotations

from pymongo import MongoClient, errors
from config import settings

class DatabaseHandler:
    """Manages the connection to the MongoDB database."""
    client: MongoClient | None = None
    db = None

    def connect_to_db(self):
        """Initializes the database connection."""
        print("Connecting to database...")
        try:
            self.client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
            self.db = self.client[settings.DB_NAME]
            # The ismaster command is cheap and does not require auth.
            self.client.admin.command('ismaster')
            print("✅ Database connected successfully.")
        except errors.ConnectionFailure as e:
            print(f"❌ Could not connect to MongoDB: {e}")
            self.client = None
            self.db = None

    def close_db_connection(self):
        """Closes the database connection."""
        if self.client:
            self.client.close()
            print("🔌 Database connection closed.")

# Create a single instance to be used across the app
db_handler = DatabaseHandler()

def get_db():
    """Dependency to get the database instance."""
    if db_handler.db is None:
        raise RuntimeError("Database is not connected.")
    return db_handler.db
