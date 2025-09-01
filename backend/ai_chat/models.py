from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union

class ChatRequest(BaseModel):
    message: str
    forecast_id: Optional[int] = None
    context_type: str = "general"  # general, specific_forecast, trend_analysis

class ChatResponse(BaseModel):
    message: str
    references: Optional[List[Dict[str, Any]]] = None
    forecast_result: Optional[Dict[str, Any]] = None
    forecast_config: Optional[Dict[str, Any]] = None
    chat_type: Optional[str] = "general"
    available_options: Optional[Dict[str, List[str]]] = None
    suggestions: Optional[List[str]] = None
