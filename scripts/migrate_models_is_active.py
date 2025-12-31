import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv(".env")

async def migrate():
    print(f"CWD: {os.getcwd()}")
    mongo_uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGO_DB_NAME", "aiagentsdb")
    
    if not mongo_uri:
        print("MONGO_URI not found in env")
        return

    client = AsyncIOMotorClient(mongo_uri)
    db = client[db_name]
    
    print(f"Connected to {db_name}")

    try:
        dbs = await client.list_database_names()
        print(f"Available databases: {dbs}")
    except Exception as e:
        print(f"Error listing databases: {e}")
    
    count = await db.ai_models.count_documents({})
    print(f"Total models found: {count}")

    result = await db.ai_models.update_many(
        {"isActive": {"$exists": False}},
        {"$set": {"isActive": True}}
    )
    
    print(f"Matched {result.matched_count} documents")
    print(f"Modified {result.modified_count} documents")

if __name__ == "__main__":
    asyncio.run(migrate())
