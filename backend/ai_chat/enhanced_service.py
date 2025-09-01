#!/usr/bin/env python3
"""
Enhanced AI chat service with comprehensive data and algorithm knowledge
"""

import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, distinct
import requests

from database import ForecastData, ExternalFactorData, SavedForecastResult, User
from .models import ChatRequest, ChatResponse
from .intelligent_forecast_generator import intelligent_forecast_generator

class EnhancedAIChatService:
    """Enhanced AI service with comprehensive forecasting knowledge"""
    
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
        
        # Algorithm information database
        self.algorithms_info = {
            "linear_regression": {
                "name": "Linear Regression",
                "description": "A statistical method that models the relationship between variables using a linear equation. Best for data with clear linear trends.",
                "use_cases": ["Simple trending data", "Baseline forecasts", "Quick analysis"],
                "pros": ["Fast computation", "Easy to interpret", "Good for linear trends"],
                "cons": ["Cannot capture non-linear patterns", "Sensitive to outliers", "Assumes linear relationship"],
                "best_for": "Data with consistent linear growth or decline patterns"
            },
            "polynomial_regression": {
                "name": "Polynomial Regression",
                "description": "Extends linear regression by fitting polynomial curves to capture non-linear relationships in data.",
                "use_cases": ["Non-linear trends", "Curved growth patterns", "Complex relationships"],
                "pros": ["Captures non-linear patterns", "Flexible curve fitting", "Good for complex trends"],
                "cons": ["Risk of overfitting", "Can be unstable", "Requires careful degree selection"],
                "best_for": "Data with curved or accelerating/decelerating trends"
            },
            "exponential_smoothing": {
                "name": "Exponential Smoothing",
                "description": "A time series forecasting technique that applies exponentially decreasing weights to historical observations.",
                "use_cases": ["Smooth trending data", "Weighted recent observations", "Simple forecasting"],
                "pros": ["Simple and fast", "Emphasizes recent data", "Good for stable trends"],
                "cons": ["No seasonality handling", "Limited for complex patterns", "Sensitive to parameter choice"],
                "best_for": "Data with stable trends where recent observations are more important"
            },
            "holt_winters": {
                "name": "Holt-Winters",
                "description": "Advanced exponential smoothing that handles both trend and seasonality components in time series data.",
                "use_cases": ["Seasonal data", "Trend + seasonality", "Retail forecasting"],
                "pros": ["Handles seasonality", "Captures trends", "Well-established method"],
                "cons": ["Requires seasonal patterns", "Parameter sensitive", "May struggle with irregular seasonality"],
                "best_for": "Data with clear seasonal patterns and trends (e.g., monthly sales cycles)"
            },
            "arima": {
                "name": "ARIMA (AutoRegressive Integrated Moving Average)",
                "description": "A sophisticated statistical model that combines autoregression, differencing, and moving averages for time series forecasting.",
                "use_cases": ["Complex time series", "Non-stationary data", "Statistical forecasting"],
                "pros": ["Statistically robust", "Handles non-stationarity", "Well-documented theory"],
                "cons": ["Complex parameter selection", "Requires expertise", "Computationally intensive"],
                "best_for": "Complex time series with autocorrelation and non-stationary behavior"
            },
            "random_forest": {
                "name": "Random Forest",
                "description": "A machine learning ensemble method that combines multiple decision trees to create robust predictions.",
                "use_cases": ["Complex patterns", "Non-linear relationships", "Feature importance"],
                "pros": ["Handles non-linearity", "Feature importance", "Robust to outliers"],
                "cons": ["Black box model", "Requires more data", "Can overfit"],
                "best_for": "Complex data with multiple influencing factors and non-linear relationships"
            },
            "seasonal_decomposition": {
                "name": "Seasonal Decomposition",
                "description": "Decomposes time series into trend, seasonal, and residual components for analysis and forecasting.",
                "use_cases": ["Seasonal analysis", "Trend extraction", "Pattern identification"],
                "pros": ["Clear component separation", "Interpretable results", "Good for analysis"],
                "cons": ["Requires clear seasonality", "Limited forecasting power", "Assumes additive components"],
                "best_for": "Understanding seasonal patterns and long-term trends in data"
            },
            "moving_average": {
                "name": "Moving Average",
                "description": "Smooths data by averaging values over a sliding window to identify trends and reduce noise.",
                "use_cases": ["Noise reduction", "Trend identification", "Simple smoothing"],
                "pros": ["Simple to understand", "Reduces noise", "Fast computation"],
                "cons": ["Lags behind trends", "No future prediction", "Loses recent information"],
                "best_for": "Smoothing noisy data and identifying underlying trends"
            },
            "sarima": {
                "name": "SARIMA (Seasonal ARIMA)",
                "description": "Extends ARIMA to handle seasonal patterns in time series data with additional seasonal parameters.",
                "use_cases": ["Seasonal time series", "Complex patterns", "Advanced forecasting"],
                "pros": ["Handles seasonality", "Statistically robust", "Flexible modeling"],
                "cons": ["Complex parameterization", "Requires expertise", "Computationally intensive"],
                "best_for": "Seasonal data requiring sophisticated statistical modeling"
            },
            "prophet_like": {
                "name": "Prophet-like Forecasting",
                "description": "Inspired by Facebook's Prophet, handles trends, seasonality, and holidays in time series forecasting.",
                "use_cases": ["Business forecasting", "Holiday effects", "Multiple seasonalities"],
                "pros": ["Handles holidays", "Multiple seasonalities", "Robust to missing data"],
                "cons": ["Requires parameter tuning", "May overfit", "Complex interpretation"],
                "best_for": "Business data with holidays, multiple seasonal patterns, and irregular events"
            },
            "lstm_like": {
                "name": "Simple LSTM-like",
                "description": "A simplified neural network approach inspired by Long Short-Term Memory networks for sequence prediction.",
                "use_cases": ["Complex patterns", "Long-term dependencies", "Non-linear forecasting"],
                "pros": ["Captures complex patterns", "Learns from sequences", "Flexible architecture"],
                "cons": ["Requires more data", "Black box", "Training intensive"],
                "best_for": "Large datasets with complex temporal dependencies and non-linear patterns"
            },
            "xgboost": {
                "name": "XGBoost",
                "description": "Extreme Gradient Boosting - a powerful machine learning algorithm that combines multiple weak learners.",
                "use_cases": ["Feature-rich data", "Non-linear patterns", "High accuracy needs"],
                "pros": ["High accuracy", "Feature importance", "Handles missing values"],
                "cons": ["Requires tuning", "Can overfit", "Less interpretable"],
                "best_for": "Complex datasets with multiple features requiring high accuracy"
            },
            "svr": {
                "name": "Support Vector Regression",
                "description": "Uses support vector machines for regression, finding optimal hyperplane for prediction.",
                "use_cases": ["Non-linear patterns", "High-dimensional data", "Robust predictions"],
                "pros": ["Handles non-linearity", "Robust to outliers", "Memory efficient"],
                "cons": ["Parameter sensitive", "Slow on large datasets", "Less interpretable"],
                "best_for": "Non-linear data where robustness to outliers is important"
            },
            "knn": {
                "name": "K-Nearest Neighbors",
                "description": "Predicts values based on the average of the k nearest historical data points.",
                "use_cases": ["Pattern matching", "Local similarities", "Simple ML approach"],
                "pros": ["Simple concept", "No training required", "Adapts to local patterns"],
                "cons": ["Sensitive to noise", "Requires good distance metric", "Memory intensive"],
                "best_for": "Data where similar historical patterns are good predictors"
            },
            "gaussian_process": {
                "name": "Gaussian Process",
                "description": "A probabilistic approach that provides uncertainty estimates along with predictions.",
                "use_cases": ["Uncertainty quantification", "Small datasets", "Smooth functions"],
                "pros": ["Uncertainty estimates", "Works with small data", "Flexible"],
                "cons": ["Computationally expensive", "Requires kernel selection", "Complex"],
                "best_for": "When uncertainty quantification is important and data is limited"
            },
            "neural_network": {
                "name": "Neural Network",
                "description": "Multi-layer perceptron that learns complex non-linear relationships through connected neurons.",
                "use_cases": ["Complex patterns", "Non-linear relationships", "Large datasets"],
                "pros": ["Learns complex patterns", "Flexible architecture", "Universal approximator"],
                "cons": ["Requires large datasets", "Black box", "Prone to overfitting"],
                "best_for": "Large datasets with complex, non-linear patterns"
            },
            "theta_method": {
                "name": "Theta Method",
                "description": "A forecasting method that decomposes time series into trend and seasonal components using theta lines.",
                "use_cases": ["Seasonal forecasting", "Trend analysis", "Competition forecasting"],
                "pros": ["Good empirical performance", "Handles seasonality", "Robust"],
                "cons": ["Limited theoretical foundation", "Parameter selection", "Complex implementation"],
                "best_for": "Seasonal data where empirical performance is prioritized"
            },
            "croston": {
                "name": "Croston's Method",
                "description": "Specialized for intermittent demand forecasting where demand occurs sporadically.",
                "use_cases": ["Intermittent demand", "Spare parts", "Low-volume items"],
                "pros": ["Handles zero demand", "Specialized for intermittent data", "Industry standard"],
                "cons": ["Only for intermittent demand", "Limited applicability", "Requires sparse data"],
                "best_for": "Spare parts, maintenance items, or products with irregular demand patterns"
            },
            "ses": {
                "name": "Simple Exponential Smoothing",
                "description": "Basic exponential smoothing for data without trend or seasonality.",
                "use_cases": ["Stable data", "No trend/seasonality", "Quick forecasting"],
                "pros": ["Very simple", "Fast computation", "Good for stable data"],
                "cons": ["No trend handling", "No seasonality", "Limited applicability"],
                "best_for": "Stable data without clear trends or seasonal patterns"
            },
            "damped_trend": {
                "name": "Damped Trend",
                "description": "Exponential smoothing with a damped trend component that prevents unrealistic long-term projections.",
                "use_cases": ["Trending data", "Conservative forecasts", "Long-term planning"],
                "pros": ["Prevents trend explosion", "Conservative estimates", "Realistic long-term forecasts"],
                "cons": ["May underestimate growth", "Limited to simple trends", "Parameter dependent"],
                "best_for": "Trending data where conservative long-term forecasts are preferred"
            },
            "naive_seasonal": {
                "name": "Naive Seasonal",
                "description": "Simple method that uses the same period from the previous season as the forecast.",
                "use_cases": ["Strong seasonality", "Baseline comparison", "Simple forecasting"],
                "pros": ["Very simple", "Good for strong seasonality", "Fast computation"],
                "cons": ["No trend handling", "Overly simplistic", "Poor for changing patterns"],
                "best_for": "Data with very strong, stable seasonal patterns and minimal trend"
            },
            "drift_method": {
                "name": "Drift Method",
                "description": "Extends naive forecasting by adding a linear trend based on historical data drift.",
                "use_cases": ["Simple trending", "Baseline forecasts", "Linear extrapolation"],
                "pros": ["Simple trend handling", "Fast computation", "Easy to understand"],
                "cons": ["Only linear trends", "No seasonality", "Overly simplistic"],
                "best_for": "Data with simple, consistent linear trends"
            },
            "best_fit": {
                "name": "Best Fit (Ensemble)",
                "description": "Automatically tests all available algorithms and selects the best performing one based on accuracy metrics.",
                "use_cases": ["Unknown patterns", "Algorithm selection", "Maximum accuracy"],
                "pros": ["Automatic selection", "Best accuracy", "Comprehensive testing"],
                "cons": ["Computationally expensive", "Takes longer", "May overfit"],
                "best_for": "When you're unsure which algorithm to use and want maximum accuracy"
            }
        }

    async def get_ai_response(self, context: str, message: str, user: User, db: Session) -> Dict[str, Any]:
        """Get enhanced AI response with comprehensive knowledge"""
        try:
            # Analyze the intent and extract entities
            intent_analysis = self._analyze_intent(message, db, user)
            
            # Build comprehensive context
            enhanced_context = self._build_enhanced_context(context, intent_analysis, db, user)
            
            # Generate AI response
            ai_response = await self._call_ollama(enhanced_context, message)
            
            # Process the response and add structured data
            processed_response = self._process_ai_response(ai_response, intent_analysis, db, user)
            
            # Handle forecast generation if detected
            if intent_analysis["forecast_request"]:
                forecast_response = await self._handle_forecast_generation(message, db, user, processed_response)
                if forecast_response:
                    processed_response.update(forecast_response)
            
            return processed_response
            
        except Exception as e:
            print(f"Error in enhanced AI service: {e}")
            return {
                "message": "I'm experiencing technical difficulties. Please ensure the AI service is running and try again.",
                "type": "error"
            }

    async def _handle_forecast_generation(self, message: str, db: Session, user: User, current_response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle automatic forecast generation from natural language"""
        try:
            if not intelligent_forecast_generator.can_generate_forecast(message):
                return None
            
            # Extract forecast configuration
            config = intelligent_forecast_generator.extract_forecast_config(message, db, user)
            if not config:
                current_response["message"] += "\n\nI'd like to generate a forecast for you, but I need more specific information. Please specify which product, customer, or location you'd like to forecast."
                return None
            
            # Generate forecast description
            forecast_description = intelligent_forecast_generator.generate_forecast_description(config)
            
            # Call the forecast API
            from main import generate_forecast_endpoint
            from pydantic import BaseModel
            
            # Create a mock request object
            class MockRequest(BaseModel):
                def __init__(self, **data):
                    super().__init__(**data)
                    for key, value in data.items():
                        setattr(self, key, value)
            
            mock_config = MockRequest(**config)
            
            # Generate the forecast
            forecast_result = await generate_forecast_endpoint(mock_config, db, user)
            
            # Update response with forecast information
            current_response["message"] += f"\n\n✅ **Forecast Generated Successfully!**\n\n{forecast_description}\n\n**Results:**\n- Accuracy: {forecast_result.get('accuracy', 0):.1f}%\n- Algorithm: {forecast_result.get('selectedAlgorithm', 'Unknown')}\n- Trend: {forecast_result.get('trend', 'Unknown').title()}"
            
            return {
                "forecast_result": forecast_result,
                "forecast_config": config,
                "type": "forecast_generated"
            }
            
        except Exception as e:
            print(f"Error generating forecast: {e}")
            current_response["message"] += f"\n\nI encountered an error while generating the forecast: {str(e)}"
            return None
    def _analyze_intent(self, message: str, db: Session, user: User) -> Dict[str, Any]:
        """Analyze user intent and extract relevant entities"""
        message_lower = message.lower()
        
        intent_analysis = {
            "intent": "general",
            "entities": {},
            "data_request": False,
            "algorithm_request": False,
            "forecast_request": False,
            "overview_request": False,
            "statistics_request": False
        }
        
        # Data-related intents
        if any(word in message_lower for word in ["data", "database", "statistics", "stats", "overview", "summary"]):
            intent_analysis["intent"] = "data_query"
            intent_analysis["data_request"] = True
            intent_analysis["statistics_request"] = True
        
        # Algorithm-related intents
        algorithm_keywords = ["algorithm", "method", "model", "technique", "approach"]
        if any(word in message_lower for word in algorithm_keywords):
            intent_analysis["intent"] = "algorithm_query"
            intent_analysis["algorithm_request"] = True
            
            # Check for specific algorithm mentions
            for algo_key, algo_info in self.algorithms_info.items():
                algo_names = [algo_info["name"].lower(), algo_key.replace("_", " ")]
                if any(name in message_lower for name in algo_names):
                    intent_analysis["entities"]["algorithm"] = algo_key
                    break
        
        # Overview/explanation requests
        overview_keywords = ["overview", "explain", "what is", "how does", "tell me about", "describe"]
        if any(keyword in message_lower for keyword in overview_keywords):
            intent_analysis["overview_request"] = True
        
        # Forecast generation intents
        forecast_keywords = ["forecast", "predict", "generate", "create", "run", "analyze"]
        if any(word in message_lower for word in forecast_keywords):
            intent_analysis["intent"] = "forecast_generation"
            intent_analysis["forecast_request"] = True
            
            # Extract entities for forecast generation
            entities = self._extract_forecast_entities(message, db)
            intent_analysis["entities"].update(entities)
        
        # Extract specific data entities
        data_entities = self._extract_data_entities(message, db)
        intent_analysis["entities"].update(data_entities)
        
        return intent_analysis

    def _extract_data_entities(self, message: str, db: Session) -> Dict[str, Any]:
        """Extract product, customer, location entities from message"""
        entities = {}
        
        # Get available options from database
        products = [p[0] for p in db.query(distinct(ForecastData.product)).filter(ForecastData.product.isnot(None)).all()]
        customers = [c[0] for c in db.query(distinct(ForecastData.customer)).filter(ForecastData.customer.isnot(None)).all()]
        locations = [l[0] for l in db.query(distinct(ForecastData.location)).filter(ForecastData.location.isnot(None)).all()]
        
        message_lower = message.lower()
        
        # Find mentioned products
        for product in products:
            if product.lower() in message_lower:
                entities["product"] = product
                break
        
        # Find mentioned customers
        for customer in customers:
            if customer.lower() in message_lower:
                entities["customer"] = customer
                break
        
        # Find mentioned locations
        for location in locations:
            if location.lower() in message_lower:
                entities["location"] = location
                break
        
        return entities

    def _extract_forecast_entities(self, message: str, db: Session) -> Dict[str, Any]:
        """Extract forecast-specific entities like time periods, algorithms"""
        entities = {}
        message_lower = message.lower()
        
        # Extract time intervals
        if "weekly" in message_lower or "week" in message_lower:
            entities["interval"] = "week"
        elif "monthly" in message_lower or "month" in message_lower:
            entities["interval"] = "month"
        elif "yearly" in message_lower or "year" in message_lower:
            entities["interval"] = "year"
        
        # Extract forecast periods
        period_match = re.search(r'(\d+)\s*(period|month|week|year)', message_lower)
        if period_match:
            entities["forecast_period"] = int(period_match.group(1))
        
        # Extract algorithm preferences
        for algo_key, algo_info in self.algorithms_info.items():
            algo_names = [algo_info["name"].lower(), algo_key.replace("_", " ")]
            if any(name in message_lower for name in algo_names):
                entities["algorithm"] = algo_key
                break
        
        # Check for "best fit" requests
        if "best fit" in message_lower or "best algorithm" in message_lower:
            entities["algorithm"] = "best_fit"
        
        return entities

    def _build_enhanced_context(self, base_context: str, intent_analysis: Dict[str, Any], db: Session, user: User) -> str:
        """Build comprehensive context for AI"""
        context_parts = [
            "You are an expert AI assistant for a comprehensive multi-variant forecasting tool.",
            "You have deep knowledge of forecasting algorithms, data analysis, and business intelligence.",
            "Provide detailed, accurate, and helpful responses based on the user's data and requirements.",
            "",
            base_context,
            ""
        ]
        
        # Add algorithm knowledge if relevant
        if intent_analysis["algorithm_request"] or intent_analysis["overview_request"]:
            context_parts.append("## ALGORITHM KNOWLEDGE:")
            context_parts.append("You have comprehensive knowledge of these forecasting algorithms:")
            
            for algo_key, algo_info in self.algorithms_info.items():
                context_parts.append(f"\n**{algo_info['name']} ({algo_key}):**")
                context_parts.append(f"- Description: {algo_info['description']}")
                context_parts.append(f"- Best for: {algo_info['best_for']}")
                context_parts.append(f"- Pros: {', '.join(algo_info['pros'])}")
                context_parts.append(f"- Cons: {', '.join(algo_info['cons'])}")
                context_parts.append(f"- Use cases: {', '.join(algo_info['use_cases'])}")
            
            context_parts.append("")
        
        # Add data statistics if relevant
        if intent_analysis["data_request"] or intent_analysis["statistics_request"]:
            data_stats = self._get_comprehensive_data_stats(db, user)
            context_parts.append("## DATABASE STATISTICS:")
            context_parts.extend(data_stats)
            context_parts.append("")
        
        # Add forecast generation guidance if relevant
        if intent_analysis["forecast_request"]:
            context_parts.append("## FORECAST GENERATION GUIDANCE:")
            context_parts.append("When generating forecasts:")
            context_parts.append("1. Use the extracted entities to configure the forecast")
            context_parts.append("2. Provide clear explanations of algorithm selection")
            context_parts.append("3. Explain the forecast results in business terms")
            context_parts.append("4. Suggest improvements or alternative approaches")
            context_parts.append("")
        
        # Add specific entity context
        if intent_analysis["entities"]:
            context_parts.append("## EXTRACTED ENTITIES:")
            for entity_type, entity_value in intent_analysis["entities"].items():
                context_parts.append(f"- {entity_type}: {entity_value}")
            context_parts.append("")
        
        return "\n".join(context_parts)

    def _get_comprehensive_data_stats(self, db: Session, user: User) -> List[str]:
        """Get comprehensive database statistics"""
        stats = []
        
        try:
            # Basic counts
            total_records = db.query(ForecastData).count()
            unique_products = db.query(distinct(ForecastData.product)).filter(ForecastData.product.isnot(None)).count()
            unique_customers = db.query(distinct(ForecastData.customer)).filter(ForecastData.customer.isnot(None)).count()
            unique_locations = db.query(distinct(ForecastData.location)).filter(ForecastData.location.isnot(None)).count()
            
            stats.append(f"Total Records: {total_records:,}")
            stats.append(f"Unique Products: {unique_products}")
            stats.append(f"Unique Customers: {unique_customers}")
            stats.append(f"Unique Locations: {unique_locations}")
            
            # Date range
            date_range = db.query(
                func.min(ForecastData.date).label('min_date'),
                func.max(ForecastData.date).label('max_date')
            ).first()
            
            if date_range and date_range.min_date and date_range.max_date:
                stats.append(f"Date Range: {date_range.min_date} to {date_range.max_date}")
            
            # Top products by volume
            top_products = db.query(
                ForecastData.product,
                func.sum(ForecastData.quantity).label('total_quantity')
            ).filter(ForecastData.product.isnot(None)).group_by(ForecastData.product).order_by(
                func.sum(ForecastData.quantity).desc()
            ).limit(5).all()
            
            if top_products:
                stats.append("Top 5 Products by Volume:")
                for product, quantity in top_products:
                    stats.append(f"  - {product}: {quantity:,.2f}")
            
            # Top customers by volume
            top_customers = db.query(
                ForecastData.customer,
                func.sum(ForecastData.quantity).label('total_quantity')
            ).filter(ForecastData.customer.isnot(None)).group_by(ForecastData.customer).order_by(
                func.sum(ForecastData.quantity).desc()
            ).limit(5).all()
            
            if top_customers:
                stats.append("Top 5 Customers by Volume:")
                for customer, quantity in top_customers:
                    stats.append(f"  - {customer}: {quantity:,.2f}")
            
            # Top locations by volume
            top_locations = db.query(
                ForecastData.location,
                func.sum(ForecastData.quantity).label('total_quantity')
            ).filter(ForecastData.location.isnot(None)).group_by(ForecastData.location).order_by(
                func.sum(ForecastData.quantity).desc()
            ).limit(5).all()
            
            if top_locations:
                stats.append("Top 5 Locations by Volume:")
                for location, quantity in top_locations:
                    stats.append(f"  - {location}: {quantity:,.2f}")
            
            # External factors
            external_factors = db.query(distinct(ExternalFactorData.factor_name)).all()
            if external_factors:
                factor_names = [f[0] for f in external_factors]
                stats.append(f"Available External Factors: {', '.join(factor_names)}")
            
            # User's saved forecasts
            saved_forecasts = db.query(SavedForecastResult).filter(SavedForecastResult.user_id == user.id).count()
            stats.append(f"Your Saved Forecasts: {saved_forecasts}")
            
        except Exception as e:
            stats.append(f"Error retrieving some statistics: {str(e)}")
        
        return stats

    async def _call_ollama(self, context: str, message: str) -> str:
        """Call Ollama API with enhanced prompt"""
        try:
            system_prompt = f"""You are an expert AI assistant for a comprehensive forecasting tool. 

{context}

IMPORTANT INSTRUCTIONS:
1. Provide detailed, accurate responses based on the context provided
2. When discussing algorithms, be specific about their strengths, weaknesses, and use cases
3. When analyzing data, provide actionable insights and recommendations
4. If generating forecasts, explain your reasoning and methodology
5. Use business-friendly language while maintaining technical accuracy
6. Always be helpful and provide practical advice
7. If you don't have enough information, clearly state what additional information you need

Respond naturally and conversationally while being informative and helpful."""

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": f"{system_prompt}\n\nUser: {message}\n\nAssistant:",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 1000
                    }
                },
                timeout=60
            )
            
            if response.status_code != 200:
                return "I'm having trouble connecting to the AI service. Please check if Ollama is running."
            
            result = response.json()
            return result.get("response", "I couldn't generate a response.")
            
        except requests.exceptions.ConnectionError:
            return "I can't connect to the AI service. Please ensure Ollama is running with the command 'ollama serve'."
        except requests.exceptions.Timeout:
            return "The AI service is taking too long to respond. Please try a simpler question or check the service."
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return "I encountered an error while processing your request. Please try again."

    def _process_ai_response(self, ai_response: str, intent_analysis: Dict[str, Any], db: Session, user: User) -> Dict[str, Any]:
        """Process AI response and add structured data"""
        response_data = {
            "message": ai_response,
            "type": intent_analysis["intent"],
            "references": [],
            "available_options": {},
            "suggestions": []
        }
        
        # Add data options for forecast generation
        if intent_analysis["forecast_request"]:
            try:
                products = [p[0] for p in db.query(distinct(ForecastData.product)).filter(ForecastData.product.isnot(None)).limit(10).all()]
                customers = [c[0] for c in db.query(distinct(ForecastData.customer)).filter(ForecastData.customer.isnot(None)).limit(10).all()]
                locations = [l[0] for l in db.query(distinct(ForecastData.location)).filter(ForecastData.location.isnot(None)).limit(10).all()]
                
                response_data["available_options"] = {
                    "products": products,
                    "customers": customers,
                    "locations": locations
                }
                
                # Add suggestions for forecast generation
                response_data["suggestions"] = [
                    "Generate a monthly forecast using best fit algorithm",
                    "Create a 6-month forecast for my top product",
                    "Run seasonal analysis on customer data",
                    "Compare different algorithms for this data"
                ]
            except Exception as e:
                print(f"Error getting options: {e}")
        
        # Add algorithm suggestions for algorithm queries
        if intent_analysis["algorithm_request"]:
            response_data["suggestions"] = [
                "Explain the difference between ARIMA and Holt-Winters",
                "Which algorithm is best for seasonal data?",
                "Show me all available algorithms",
                "What's the most accurate algorithm for my data?"
            ]
        
        # Add data exploration suggestions for data queries
        if intent_analysis["data_request"]:
            response_data["suggestions"] = [
                "Show me my top 5 products by volume and their trends",
                "Analyze my customer purchasing patterns",
                "What seasonal patterns exist in my data?",
                "Which locations show the strongest growth?",
                "Compare performance across different product categories"
            ]
        
        return response_data

    def get_algorithm_overview(self, algorithm_key: str) -> Dict[str, Any]:
        """Get comprehensive algorithm overview"""
        if algorithm_key not in self.algorithms_info:
            return {
                "error": f"Algorithm '{algorithm_key}' not found",
                "available_algorithms": list(self.algorithms_info.keys())
            }
        
        algo_info = self.algorithms_info[algorithm_key]
        return {
            "algorithm": algorithm_key,
            "overview": algo_info,
            "recommendations": self._get_algorithm_recommendations(algorithm_key),
            "related_algorithms": self._get_related_algorithms(algorithm_key)
        }

    def _get_algorithm_recommendations(self, algorithm_key: str) -> List[str]:
        """Get recommendations for when to use this algorithm"""
        recommendations_map = {
            "linear_regression": [
                "Use when your data shows a clear linear trend",
                "Good for quick baseline forecasts",
                "Ideal for simple, predictable growth patterns"
            ],
            "random_forest": [
                "Use when you have multiple influencing factors",
                "Good for complex, non-linear patterns",
                "Ideal when you need feature importance analysis"
            ],
            "holt_winters": [
                "Use for data with clear seasonal patterns",
                "Good for retail, sales, or cyclical business data",
                "Ideal when both trend and seasonality are present"
            ],
            "best_fit": [
                "Use when you're unsure which algorithm to choose",
                "Good for maximizing accuracy across different data types",
                "Ideal for comprehensive analysis and comparison"
            ]
        }
        
        return recommendations_map.get(algorithm_key, [
            "Consult the algorithm description for specific use cases",
            "Consider your data characteristics when choosing",
            "Test with your specific dataset for best results"
        ])

    def _get_related_algorithms(self, algorithm_key: str) -> List[str]:
        """Get algorithms related to the specified one"""
        algorithm_families = {
            "statistical": ["linear_regression", "polynomial_regression", "exponential_smoothing", "holt_winters", "arima", "sarima"],
            "machine_learning": ["random_forest", "xgboost", "neural_network", "svr", "knn", "gaussian_process"],
            "simple": ["moving_average", "ses", "naive_seasonal", "drift_method"],
            "specialized": ["croston", "theta_method", "prophet_like", "lstm_like"]
        }
        
        for family, algorithms in algorithm_families.items():
            if algorithm_key in algorithms:
                return [alg for alg in algorithms if alg != algorithm_key]
        
        return []

    def get_data_insights(self, db: Session, user: User, entity_filters: Dict[str, str] = None) -> Dict[str, Any]:
        """Get comprehensive data insights"""
        insights = {
            "summary": {},
            "trends": {},
            "recommendations": []
        }
        
        try:
            # Build query with filters
            query = db.query(ForecastData)
            
            if entity_filters:
                if entity_filters.get("product"):
                    query = query.filter(ForecastData.product == entity_filters["product"])
                if entity_filters.get("customer"):
                    query = query.filter(ForecastData.customer == entity_filters["customer"])
                if entity_filters.get("location"):
                    query = query.filter(ForecastData.location == entity_filters["location"])
            
            # Get basic statistics
            total_quantity = query.with_entities(func.sum(ForecastData.quantity)).scalar() or 0
            avg_quantity = query.with_entities(func.avg(ForecastData.quantity)).scalar() or 0
            record_count = query.count()
            
            insights["summary"] = {
                "total_quantity": float(total_quantity),
                "average_quantity": float(avg_quantity),
                "record_count": record_count
            }
            
            # Get date range
            date_stats = query.with_entities(
                func.min(ForecastData.date).label('min_date'),
                func.max(ForecastData.date).label('max_date')
            ).first()
            
            if date_stats and date_stats.min_date and date_stats.max_date:
                insights["summary"]["date_range"] = {
                    "start": date_stats.min_date.isoformat(),
                    "end": date_stats.max_date.isoformat()
                }
            
            # Generate recommendations
            if record_count < 12:
                insights["recommendations"].append("Consider collecting more historical data for better forecast accuracy")
            
            if avg_quantity > 1000:
                insights["recommendations"].append("High-volume data detected - Random Forest or XGBoost may perform well")
            
            insights["recommendations"].append("Use Best Fit algorithm to automatically find the most suitable method")
            
        except Exception as e:
            insights["error"] = str(e)
        
        return insights

# Global service instance
enhanced_ai_service = EnhancedAIChatService()

async def get_ai_response(context: str, message: str, user: User, db: Session) -> Dict[str, Any]:
    """Main entry point for AI responses"""
    return await enhanced_ai_service.get_ai_response(context, message, user, db)