from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    message: str
    forecast_id: Optional[int] = None
    context_type: str = "general"  # general, specific_forecast, trend_analysis

class ChatResponse(BaseModel):
    message: str
    references: Optional[List[Dict[str, Any]]] = None
