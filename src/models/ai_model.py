from pydantic import BaseModel, Field
from typing import Optional
from bson import ObjectId

class AIModelCreate(BaseModel):
    """Model for creating AI Model"""
    name: str
    description: Optional[str] = None
    model_id: str  # e.g., "gemini/gemini-2.0-flash-thinking-exp-1219"
    api_key: str
    base_url: Optional[str] = "https://generativelanguage.googleapis.com/v1beta"  # Default to Gemini
    isActive: bool = True  # Default to True for new models
    cost: dict = Field(default_factory=lambda: {"input_per_million": 0, "output_per_million": 0})
    limits: dict = Field(default_factory=lambda: {
        "max_input_tokens": None, 
        "max_output_tokens": None,
        "input_types": ["text"],
        "output_types": ["text"]
    })
    details: dict = Field(default_factory=lambda: {"requests_per_day": None, "requests_per_minute": None})

class AIModel(AIModelCreate):
    """Full AI Model with database fields"""
    id: Optional[str] = Field(None, alias="_id")
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "name": "Gemini Flash Thinking",
                "model_id": "gemini/gemini-2.0-flash-thinking-exp-1219",
                "api_key": "your-api-key",
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
                "cost": {
                    "input_per_million": 1.5,
                    "output_per_million": 2.0
                },
                "limits": {
                    "max_input_tokens": 32000,
                    "max_output_tokens": 8000,
                    "input_types": ["text", "image"],
                    "output_types": ["text"]
                },
                "details": {
                    "requests_per_day": 1500,
                    "requests_per_minute": 6
                }
            }
        }
