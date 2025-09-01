from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from database import get_db, SavedForecastResult, User
from auth import get_current_user
from .models import ChatRequest, ChatResponse
from .enhanced_service import get_ai_response
from .utils import build_context_from_db

router = APIRouter(prefix="/ai", tags=["AI Chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Enhanced AI chat endpoint with comprehensive forecasting knowledge"""
    # Build context from database
    context = build_context_from_db(db, request, current_user)
    
    # Get response from AI service
    ai_response = await get_ai_response(context, request.message, current_user, db)
    
    return ChatResponse(
        message=ai_response.get("message", ai_response.get("content", "I couldn't process your request.")),
        references=ai_response.get("references"),
        forecast_result=ai_response.get("forecast_result"),
        forecast_config=ai_response.get("forecast_config"),
        chat_type=ai_response.get("type", "general"),
        available_options=ai_response.get("available_options"),
        suggestions=ai_response.get("suggestions")
    )

@router.get("/algorithms")
async def get_algorithms_info():
    """Get comprehensive information about all available algorithms"""
    from .enhanced_service import enhanced_ai_service
    return {
        "algorithms": enhanced_ai_service.algorithms_info,
        "total_count": len(enhanced_ai_service.algorithms_info)
    }

@router.get("/algorithms/{algorithm_key}")
async def get_algorithm_overview(algorithm_key: str):
    """Get detailed overview of a specific algorithm"""
    from .enhanced_service import enhanced_ai_service
    return enhanced_ai_service.get_algorithm_overview(algorithm_key)

@router.get("/data_insights")
async def get_data_insights(
    product: str = None,
    customer: str = None,
    location: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive data insights with optional filtering"""
    from .enhanced_service import enhanced_ai_service
    
    entity_filters = {}
    if product:
        entity_filters["product"] = product
    if customer:
        entity_filters["customer"] = customer
    if location:
        entity_filters["location"] = location
    
    return enhanced_ai_service.get_data_insights(db, current_user, entity_filters)

@router.get("/health")
async def ai_health_check():
    """Check AI service health"""
    from .health_check import AIHealthChecker
    return AIHealthChecker.get_comprehensive_health()