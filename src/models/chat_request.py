from typing import Any, Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    messages: list[dict[str, Any]]
    model_id: Optional[str] = None
    enable_grounding: Optional[bool] = False