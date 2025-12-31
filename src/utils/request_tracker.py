from src.config.db import get_database
from src.models.request_log import RequestLog
from src.utils.logger import logger

async def track_request(log_data: dict):
    """
    Track LLM request by inserting log entry into MongoDB
    
    Args:
        log_data: Dictionary containing request log data
    """
    try:
        db = get_database()
        
        # Skip tracking if database is not connected
        if db is None:
            logger.debug("Skipping request tracking - MongoDB not connected")
            return
        
        # Create RequestLog model for validation
        request_log = RequestLog(**log_data)
        
        # Insert into MongoDB
        collection = db.request_logs
        await collection.insert_one(request_log.model_dump())
        
        logger.debug(f"Tracked request: {request_log.ai_model_id} - {request_log.total_tokens} tokens in {request_log.response_time_ms}ms")
    
    except Exception as e:
        # Don't let tracking failures break the main request
        logger.error(f"Failed to track request: {str(e)}")
