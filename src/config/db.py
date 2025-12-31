import os
from motor.motor_asyncio import AsyncIOMotorClient
from src.utils.logger import logger

# MongoDB client and database instances
mongo_client = None
db = None

async def connect_to_mongo():
    """Initialize MongoDB connection"""
    global mongo_client, db
    
    try:
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            logger.warning("MONGODB_URI not found in environment variables. Request tracking will be disabled.")
            return None
        
        mongo_client = AsyncIOMotorClient(mongodb_uri)
        # Test the connection
        await mongo_client.admin.command('ping')
        
        # Get database name from URI or use default
        db = mongo_client.get_database()
        
        logger.info(f"MongoDB Connected: {mongo_client.address[0]}")
        return db
    
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

async def close_mongo_connection():
    """Close MongoDB connection"""
    global mongo_client
    
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed")

def get_database():
    """Get database instance"""
    return db
