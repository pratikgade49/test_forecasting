from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from database import get_db, SavedForecastResult, User
from auth import get_current_user
from .models import ChatRequest, ChatResponse
from .service import get_ai_response
from .utils import build_context_from_db

router = APIRouter(prefix="/ai", tags=["AI Chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Build context from database
    context = build_context_from_db(db, request, current_user)
    
    # Get response from AI service
    ai_response = await get_ai_response(context, request.message)
    
    return ChatResponse(
        message=ai_response["content"],
        references=ai_response.get("references")
    )
