from datetime import datetime
from bson import ObjectId
from src.config.db import get_database
from src.utils.logger import logger

async def check_grounding_limit(model_id: str, limit_type: str = "websearch") -> bool:
    """
    Check if the model has exceeded its grounding limit for the day.
    
    Args:
        model_id: The ID of the AI model.
        limit_type: The type of grounding limit to check (default: "websearch").
        
    Returns:
        bool: True if limit is NOT exceeded (allowed), False if exceeded.
    """
    try:
        db = get_database()
        if db is None:
            logger.warning("Database not connected, skipping limit check (failing open)")
            return True
            
        # Get the model to find the limit
        model = await db.ai_models.find_one({"_id": ObjectId(model_id)})
        if not model:
            logger.warning(f"Model {model_id} not found during limit check")
            return True # Fail open if model not found
            
        groundings = model.get("groundings", {})
        if not groundings:
            return True # No limits defined
            
        limit = groundings.get(limit_type)
        if limit is None or limit == 0:
            return True # No specific limit for this type
            
        # Count usage for today
        # We need to count request logs for this model, where grounding_enabled is True, since midnight
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        count = await db.request_logs.count_documents({
            "ai_model_id": ObjectId(model_id),
            "grounding_enabled": True,
            "timestamp": {"$gte": today_start}
        })
        
        if count >= limit:
            logger.warning(f"Grounding limit exceeded for model {model.get('name')} (ID: {model_id}). Limit: {limit}, Used: {count}")
            return False
            
        return True

    except Exception as e:
        logger.error(f"Error checking grounding limit: {str(e)}")
        # Start fail open policy for resilience, but log error
        return True
