from sqlalchemy.orm import Session
from typing import Dict, Any, List
import json

from database import SavedForecastResult, ForecastData, User
from .models import ChatRequest

def build_context_from_db(db: Session, request: ChatRequest, user: User) -> str:
    """Build context for AI from database data"""
    context_parts = [
        "You are an AI assistant for a multivariant forecasting tool. You help users understand their forecast data, trends, and metrics.",
        "Answer questions based on the data provided in this context. If you don't have enough information, say so clearly.",
        "When discussing metrics, be precise and explain what they mean in business terms."
    ]
    
    # Add user-specific context
    context_parts.append(f"You are speaking with user: {user.username}")
    
    if request.forecast_id:
        # Add specific forecast context
        forecast = db.query(SavedForecastResult).filter(
            SavedForecastResult.id == request.forecast_id,
            SavedForecastResult.user_id == user.id
        ).first()
        
        if forecast:
            context_parts.append("\n## FORECAST INFORMATION:")
            context_parts.append(f"Forecast Name: {forecast.name}")
            context_parts.append(f"Description: {forecast.description or 'No description provided'}")
            
            # Parse JSON fields stored as text
            config_dict = None
            result_dict = None
            try:
                config_dict = json.loads(forecast.forecast_config) if isinstance(forecast.forecast_config, str) else forecast.forecast_config
            except Exception:
                config_dict = None
            try:
                result_dict = json.loads(forecast.forecast_data) if isinstance(forecast.forecast_data, str) else forecast.forecast_data
            except Exception:
                result_dict = None

            if isinstance(config_dict, dict):
                context_parts.append(f"Forecast By: {config_dict.get('forecastBy')}")
                context_parts.append(f"Algorithm: {config_dict.get('algorithm')}")
                context_parts.append(f"Interval: {config_dict.get('interval')}")
                context_parts.append(f"Historic Period: {config_dict.get('historicPeriod')}")
                context_parts.append(f"Forecast Period: {config_dict.get('forecastPeriod')}")

            # Add forecast results summary
            if isinstance(result_dict, dict):
                # Single or multi result
                if 'accuracy' in result_dict:
                    context_parts.append(f"Accuracy: {result_dict.get('accuracy')}")
                    context_parts.append(f"MAE: {result_dict.get('mae')}")
                    context_parts.append(f"RMSE: {result_dict.get('rmse')}")
                    if result_dict.get('trend'):
                        context_parts.append(f"Trend: {result_dict.get('trend')}")
                elif 'results' in result_dict and isinstance(result_dict['results'], list) and len(result_dict['results']) > 0:
                    best = max(result_dict['results'], key=lambda r: r.get('accuracy', 0))
                    context_parts.append(f"Best Combination Accuracy: {best.get('accuracy')}")
                    if best.get('trend'):
                        context_parts.append(f"Trend: {best.get('trend')}")
    
    # Add general database stats for context
    data_count = db.query(ForecastData).count()
    context_parts.append(f"\nThe database contains {data_count} historical data points.")
    
    # Combine all context parts
    return "\n".join(context_parts)
