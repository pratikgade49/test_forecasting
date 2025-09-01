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

class AlgorithmInfo(BaseModel):
    name: str
    description: str
    use_cases: List[str]
    pros: List[str]
    cons: List[str]
    best_for: str

class DataInsights(BaseModel):
    summary: Dict[str, Any]
    trends: Dict[str, Any]
    recommendations: List[str]

class HealthStatus(BaseModel):
    overall_status: str
    services: Dict[str, Dict[str, Any]]
    recommendations: List[str]