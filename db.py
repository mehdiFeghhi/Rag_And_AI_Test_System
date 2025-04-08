from motor.motor_asyncio import AsyncIOMotorClient
import os

# MongoDB connection string (adjust as needed)
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URL)
db = client.chat_app  # Your database
chat_collection = db.chats  # Collection to store chats
