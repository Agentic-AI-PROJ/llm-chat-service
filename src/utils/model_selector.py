from src.config.db import get_database
from src.utils.logger import logger
from typing import Optional, Dict
from datetime import datetime, timedelta
import asyncio

class ModelSelector:
    """
    Round-robin model selector for distributing LLM requests across multiple models.
    Helps avoid rate limiting by rotating through available models and checking RPM/RPD limits.
    """
    
    def __init__(self):
        self.models = []
        self.current_index = 0
        self.lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self):
        """Load models from database on startup"""
        try:
            await self.refresh_models()
            self._initialized = True
            logger.info(f"ModelSelector initialized with {len(self.models)} models")
        except Exception as e:
            logger.error(f"Failed to initialize ModelSelector: {str(e)}")
            self._initialized = False
    
    async def refresh_models(self):
        """Refresh the list of available models from database"""
        try:
            db = get_database()
            if db is None:
                logger.warning("Database not connected, cannot refresh models")
                return
            
            models = []
            # Only fetch active models
            query = {"isActive": True}
            async for model in db.ai_models.find(query):
                models.append(model)
            
            async with self.lock:
                self.models = models
                # Reset index if models list changed
                if self.current_index >= len(self.models):
                    self.current_index = 0
            
            logger.debug(f"Refreshed models: {len(models)} available")
        
        except Exception as e:
            logger.error(f"Error refreshing models: {str(e)}")
    
    async def check_rate_limits(self, model: Dict) -> bool:
        """
        Check if model has exceeded its RPM or RPD limits.
        Returns True if model is available, False if rate-limited.
        """
        try:
            db = get_database()
            if db is None:
                return True  # If no DB, allow the model
            
            model_id = model["_id"]
            now = datetime.utcnow()
            
            # Get RPM and RPD limits
            rpm_limit = model.get("details", {}).get("requests_per_minute")
            rpd_limit = model.get("details", {}).get("requests_per_day")
            
            # If no limits defined, allow
            if not rpm_limit and not rpd_limit:
                return True
            
            # Check RPM (requests in last minute)
            if rpm_limit:
                one_minute_ago = now - timedelta(minutes=1)
                rpm_count = await db.request_logs.count_documents({
                    "ai_model_id": model_id,
                    "timestamp": {"$gte": one_minute_ago}
                })
                if rpm_count >= rpm_limit:
                    logger.warning(f"Model {model.get('name')} hit RPM limit: {rpm_count}/{rpm_limit}")
                    return False
            
            # Check RPD (requests in last 24 hours)
            if rpd_limit:
                one_day_ago = now - timedelta(days=1)
                rpd_count = await db.request_logs.count_documents({
                    "ai_model_id": model_id,
                    "timestamp": {"$gte": one_day_ago}
                })
                if rpd_count >= rpd_limit:
                    logger.warning(f"Model {model.get('name')} hit RPD limit: {rpd_count}/{rpd_limit}")
                    return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error checking rate limits: {str(e)}")
            return True  # On error, allow the model
    
    async def get_next_model(self, required_input_type: str = "text", model_id: Optional[str] = None) -> Optional[Dict]:
        """
        Get the next available model in round-robin order that hasn't hit rate limits
        and supports the required input type.
        If model_id is provided, only considers models with that specific ID.
        Returns None if no models are available.
        """
        if not self._initialized:
            await self.initialize()
        
        async with self.lock:
            if not self.models:
                logger.warning("No models available for selection")
                return None
            
            # Try each model in round-robin order
            # We iterate max len(models) times to check everyone.
            # But since we use a shared index, we want to start from current_index.
            start_index = self.current_index
            
            for _ in range(len(self.models)):
                model = self.models[self.current_index]
                
                # Increment index for next call (circular)
                self.current_index = (self.current_index + 1) % len(self.models)
                
                # Check if model matches requested model_id
                if model_id and model.get("model_id") != model_id:
                    continue
                
                # Check if model supports required input type
                start_input_types = model.get("limits", {}).get("input_types", ["text"])
                if required_input_type not in start_input_types:
                    logger.debug(f"Skipping model {model.get('name')} (doesn't support {required_input_type})")
                    continue

                # Check if model is within rate limits
                if await self.check_rate_limits(model):
                    logger.debug(f"Selected model: {model.get('name')} (index {(self.current_index - 1) % len(self.models)}/{len(self.models) - 1})")
                    return model
                else:
                    logger.debug(f"Skipping rate-limited model: {model.get('name')}")
            
            # All available models checked and none matched or all rate-limited
            logger.error(f"No models available for {required_input_type}" + (f" with model_id {model_id}" if model_id else ""))
            return None
    
    def is_initialized(self) -> bool:
        """Check if selector has been initialized"""
        return self._initialized

# Global instance
model_selector = ModelSelector()
