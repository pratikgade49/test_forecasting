from sqlalchemy.orm import Session
from typing import Dict, Any, List
import json

from database import SavedForecastResult, ForecastData, User
from .models import ChatRequest

def build_context_from_db(db: Session, request: ChatRequest, user: User) -> str:
    """Build context for AI from database data"""
    context_parts = [
        "You are an expert AI assistant for a comprehensive multi-variant forecasting tool.",
        "You have deep knowledge of 23+ forecasting algorithms, data analysis, and business intelligence.",
        "You can help users with:",
        "- Understanding their data (products, customers, locations, trends)",
        "- Explaining forecasting algorithms and their use cases",
        "- Generating forecasts through natural language",
        "- Analyzing forecast results and providing insights",
        "- Recommending best practices for forecasting",
        "",
        "Always provide detailed, accurate, and actionable responses."
    ]
    
    # Add user-specific context
    context_parts.append(f"You are speaking with user: {user.username}")
    
    # Add comprehensive database overview
    try:
        # Get database statistics
        total_records = db.query(ForecastData).count()
        if total_records > 0:
            context_parts.append(f"\n## DATABASE OVERVIEW:")
            context_parts.append(f"Total records in database: {total_records:,}")
            
            # Get unique counts
            from sqlalchemy import distinct, func
            unique_products = db.query(distinct(ForecastData.product)).filter(ForecastData.product.isnot(None)).count()
            unique_customers = db.query(distinct(ForecastData.customer)).filter(ForecastData.customer.isnot(None)).count()
            unique_locations = db.query(distinct(ForecastData.location)).filter(ForecastData.location.isnot(None)).count()
            
            context_parts.append(f"Unique products: {unique_products}")
            context_parts.append(f"Unique customers: {unique_customers}")
            context_parts.append(f"Unique locations: {unique_locations}")
            
            # Get date range
            date_range = db.query(
                func.min(ForecastData.date).label('min_date'),
                func.max(ForecastData.date).label('max_date')
            ).first()
            
            if date_range and date_range.min_date and date_range.max_date:
                context_parts.append(f"Date range: {date_range.min_date} to {date_range.max_date}")
            
            # Get top products by volume
            top_products = db.query(
                ForecastData.product,
                func.sum(ForecastData.quantity).label('total_quantity')
            ).filter(ForecastData.product.isnot(None)).group_by(ForecastData.product).order_by(
                func.sum(ForecastData.quantity).desc()
            ).limit(5).all()
            
            if top_products:
                context_parts.append("Top 5 products by volume:")
                for product, quantity in top_products:
                    context_parts.append(f"  - {product}: {quantity:,.2f}")
            
            # Get external factors if available
            from database import ExternalFactorData
            external_factors = db.query(distinct(ExternalFactorData.factor_name)).limit(10).all()
            if external_factors:
                factor_names = [f[0] for f in external_factors]
                context_parts.append(f"Available external factors: {', '.join(factor_names)}")
        else:
            context_parts.append("\n## DATABASE STATUS:")
            context_parts.append("No data has been uploaded yet. User needs to upload data before forecasting.")
    
    except Exception as e:
        context_parts.append(f"\n## DATABASE ERROR:")
        context_parts.append(f"Error accessing database: {str(e)}")
    
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
    
    # Add user's saved forecasts context
    try:
        saved_forecasts_count = db.query(SavedForecastResult).filter(SavedForecastResult.user_id == user.id).count()
        if saved_forecasts_count > 0:
            context_parts.append(f"\nUser has {saved_forecasts_count} saved forecasts available.")
            
            # Get recent forecasts
            recent_forecasts = db.query(SavedForecastResult).filter(
                SavedForecastResult.user_id == user.id
            ).order_by(SavedForecastResult.created_at.desc()).limit(3).all()
            
            if recent_forecasts:
                context_parts.append("Recent forecasts:")
                for forecast in recent_forecasts:
                    context_parts.append(f"  - {forecast.name}")
    except Exception as e:
        print(f"Error getting saved forecasts context: {e}")
    
    # Combine all context parts
    return "\n".join(context_parts)
