from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional
from bson import ObjectId

class RequestLog(BaseModel):
    """Model for LLM request log tracking"""
    ai_model_id: ObjectId  # AIModel ObjectId reference
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    response_time_ms: float
    success: bool
    grounding_enabled: Optional[bool] = False
    input_text: Optional[str] = None
    output_text: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('ai_model_id', mode='before')
    @classmethod
    def validate_object_id(cls, v):
        if isinstance(v, str):
            return ObjectId(v)
        return v
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str
        }
        json_schema_extra = {
            "example": {
                "ai_model_id": "507f1f77bcf86cd799439011",
                "input_tokens": 150,
                "output_tokens": 300,
                "total_tokens": 450,
                "response_time_ms": 1234.56,
                "success": True,
                "timestamp": "2025-12-17T15:23:00Z"
            }
        }
