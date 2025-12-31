from fastapi import APIRouter, HTTPException
from src.models.ai_model import AIModel, AIModelCreate
from src.config.db import get_database
from src.utils.logger import logger
from bson import ObjectId
from datetime import datetime

router = APIRouter()

@router.post("", response_model=dict)
async def create_model(model: AIModelCreate):
    """Create a new AI model"""
    try:
        db = get_database()
        if db is None:
            raise HTTPException(status_code=503, detail="Database not connected")
        
        model_data = model.model_dump()
        model_data["created_at"] = datetime.utcnow().isoformat()
        model_data["updated_at"] = datetime.utcnow().isoformat()
        
        result = await db.ai_models.insert_one(model_data)
        model_data["_id"] = str(result.inserted_id)
        
        return {"success": True, "data": model_data}
    
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/active", response_model=dict)
async def get_active_models():
    """Get all active AI models"""
    try:
        db = get_database()
        if db is None:
            raise HTTPException(status_code=503, detail="Database not connected")
        
        models = []
        # Find all active models
        async for model in db.ai_models.find({"isActive": True}):
            model["_id"] = str(model["_id"])
            models.append(model)
        
        return {"success": True, "data": models}
    
    except Exception as e:
        logger.error(f"Error fetching active models: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("", response_model=dict)
async def get_models():
    """Get all AI models"""
    try:
        db = get_database()
        if db is None:
            raise HTTPException(status_code=503, detail="Database not connected")
        
        models = []
        async for model in db.ai_models.find():
            model["_id"] = str(model["_id"])
            models.append(model)
        
        return {"success": True, "data": models}
    
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{model_id}", response_model=dict)
async def get_model(model_id: str):
    """Get a specific AI model by ID"""
    try:
        db = get_database()
        if db is None:
            raise HTTPException(status_code=503, detail="Database not connected")
        
        model = await db.ai_models.find_one({"_id": ObjectId(model_id)})
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model["_id"] = str(model["_id"])
        return {"success": True, "data": model}
    
    except Exception as e:
        logger.error(f"Error fetching model: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/{model_id}", response_model=dict)
async def update_model(model_id: str, model_update: AIModelCreate):
    """Update an AI model"""
    try:
        db = get_database()
        if db is None:
            raise HTTPException(status_code=503, detail="Database not connected")
        
        update_data = model_update.model_dump()
        update_data["updated_at"] = datetime.utcnow().isoformat()
        
        result = await db.ai_models.update_one(
            {"_id": ObjectId(model_id)},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Model not found")
        
        updated_model = await db.ai_models.find_one({"_id": ObjectId(model_id)})
        updated_model["_id"] = str(updated_model["_id"])
        
        return {"success": True, "data": updated_model}
    
    except Exception as e:
        logger.error(f"Error updating model: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{model_id}", response_model=dict)
async def delete_model(model_id: str):
    """Delete an AI model"""
    try:
        db = get_database()
        if db is None:
            raise HTTPException(status_code=503, detail="Database not connected")
        
        result = await db.ai_models.delete_one({"_id": ObjectId(model_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {"success": True, "message": "Model deleted successfully"}
    
    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
