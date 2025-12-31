import asyncio
import os
import sys
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# Add project root to path to import config if needed, 
# though we'll just use motor directly here for simplicity script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def migrate():
    load_dotenv()
    
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        print("Error: MONGODB_URI not found in environment variables")
        return

    client = AsyncIOMotorClient(mongo_uri)
    db = client.get_database()
    collection = db["ai_models"]
    
    print(f"Connected to database: {db.name}")
    
    cursor = collection.find({})
    updated_count = 0
    
    async for model in cursor:
        print(f"Processing model: {model.get('name')} ({model.get('model_id')})")
        
        limits = model.get("limits", {})
        update_needed = False
        
        # Determine defaults based on model ID keywords
        model_id = model.get("model_id", "").lower()
        default_inputs = ["text"]
        
        # Simple heuristic for existing models that might support images
        if "pro" in model_id or "flash" in model_id or "gpt-4o" in model_id or "vision" in model_id:
             if "gemini" in model_id or "gpt" in model_id:
                 default_inputs = ["text", "image"]
        
        if "input_types" not in limits:
            limits["input_types"] = default_inputs
            update_needed = True
            print(f"  - Adding input_types: {default_inputs}")
            
        if "output_types" not in limits:
            limits["output_types"] = ["text"]
            update_needed = True
            print(f"  - Adding output_types: ['text']")
            
        if update_needed:
            await collection.update_one(
                {"_id": model["_id"]},
                {"$set": {"limits": limits}}
            )
            updated_count += 1
            print("  - Updated.")
        else:
            print("  - No changes needed.")
            
    print(f"\nMigration complete. Updated {updated_count} models.")
    client.close()

if __name__ == "__main__":
    asyncio.run(migrate())
