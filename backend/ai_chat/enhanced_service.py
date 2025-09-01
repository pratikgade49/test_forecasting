import os
import json
import re
from typing import Dict, Any, List, Optional
import requests
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database import get_db, ForecastData, SavedForecastResult, User
from model_persistence import ModelPersistenceManager

class ForecastChatService:
    """Enhanced AI chat service for forecast scheduling and generation"""
    
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
        
    async def process_chat_message(self, message: str, user: User, db: Session, context: str = "") -> Dict[str, Any]:
        """Process chat message and determine if it's a forecast request"""
        
        # Analyze the message to determine intent
        intent = self._analyze_intent(message)
        
        if intent["type"] == "forecast_request":
            return await self._handle_forecast_request(message, intent, user, db)
        elif intent["type"] == "schedule_request":
            return await self._handle_schedule_request(message, intent, user, db)
        elif intent["type"] == "data_query":
            return await self._handle_data_query(message, intent, user, db, context)
        else:
            return await self._handle_general_chat(message, context)
    
    def _analyze_intent(self, message: str) -> Dict[str, Any]:
        """Analyze message to determine user intent"""
        message_lower = message.lower()
        
        # Forecast generation keywords
        forecast_keywords = [
            "forecast", "predict", "generate forecast", "create forecast", 
            "run forecast", "analyze", "prediction", "future sales"
        ]
        
        # Scheduling keywords
        schedule_keywords = [
            "schedule", "set up", "automate", "recurring", "daily", 
            "weekly", "monthly", "every", "remind me"
        ]
        
        # Data query keywords
        data_keywords = [
            "show me", "what is", "how much", "total", "average", 
            "data", "records", "sales", "performance"
        ]
        
        # Extract entities (products, customers, locations, time periods)
        entities = self._extract_entities(message)
        
        if any(keyword in message_lower for keyword in forecast_keywords):
            return {
                "type": "forecast_request",
                "entities": entities,
                "confidence": 0.9
            }
        elif any(keyword in message_lower for keyword in schedule_keywords):
            return {
                "type": "schedule_request", 
                "entities": entities,
                "confidence": 0.8
            }
        elif any(keyword in message_lower for keyword in data_keywords):
            return {
                "type": "data_query",
                "entities": entities,
                "confidence": 0.7
            }
        else:
            return {
                "type": "general_chat",
                "entities": entities,
                "confidence": 0.5
            }
    
    def _extract_entities(self, message: str) -> Dict[str, Any]:
        """Extract entities like products, time periods, algorithms from message"""
        entities = {
            "products": [],
            "customers": [],
            "locations": [],
            "time_period": None,
            "algorithm": None,
            "interval": None
        }
        
        # Extract time periods
        time_patterns = {
            "week": ["week", "weekly", "7 days"],
            "month": ["month", "monthly", "30 days"],
            "year": ["year", "yearly", "annual", "12 months"]
        }
        
        for interval, patterns in time_patterns.items():
            if any(pattern in message.lower() for pattern in patterns):
                entities["interval"] = interval
                break
        
        # Extract algorithm preferences
        algorithm_patterns = {
            "linear_regression": ["linear", "simple", "basic"],
            "random_forest": ["random forest", "machine learning", "ml"],
            "best_fit": ["best", "automatic", "auto", "optimal"],
            "arima": ["arima", "time series"],
            "holt_winters": ["seasonal", "holt", "winters"]
        }
        
        for algorithm, patterns in algorithm_patterns.items():
            if any(pattern in message.lower() for pattern in patterns):
                entities["algorithm"] = algorithm
                break
        
        # Extract numbers for periods
        numbers = re.findall(r'\b(\d+)\b', message)
        if numbers:
            entities["periods"] = [int(n) for n in numbers]
        
        return entities
    
    async def _handle_forecast_request(self, message: str, intent: Dict, user: User, db: Session) -> Dict[str, Any]:
        """Handle forecast generation requests"""
        try:
            # Get available options from database
            products = db.query(ForecastData.product).distinct().all()
            customers = db.query(ForecastData.customer).distinct().all()
            locations = db.query(ForecastData.location).distinct().all()
            
            product_list = [p[0] for p in products if p[0]]
            customer_list = [c[0] for c in customers if c[0]]
            location_list = [l[0] for l in locations if l[0]]
            
            # Try to match entities with available data
            matched_entities = self._match_entities_with_data(
                intent["entities"], product_list, customer_list, location_list
            )
            
            if not matched_entities["found_matches"]:
                # Ask for clarification
                return {
                    "message": self._generate_clarification_request(product_list, customer_list, location_list),
                    "type": "clarification_needed",
                    "available_options": {
                        "products": product_list[:10],  # Limit for readability
                        "customers": customer_list[:10],
                        "locations": location_list[:10]
                    }
                }
            
            # Generate forecast configuration
            config = self._create_forecast_config(matched_entities, intent["entities"])
            
            # Generate the forecast
            from main import ForecastingEngine
            result = ForecastingEngine.generate_forecast(db, config)
            
            # Auto-save the forecast
            auto_save_name = f"AI Chat Forecast {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            saved_forecast = SavedForecastResult(
                user_id=user.id,
                name=auto_save_name,
                description=f"Generated via AI chat: {message[:100]}...",
                forecast_config=json.dumps(config.dict()),
                forecast_data=json.dumps(result.dict())
            )
            db.add(saved_forecast)
            db.commit()
            
            return {
                "message": self._format_forecast_response(result, config),
                "type": "forecast_generated",
                "forecast_result": result.dict(),
                "forecast_config": config.dict(),
                "saved_forecast_id": saved_forecast.id
            }
            
        except Exception as e:
            return {
                "message": f"I encountered an error generating the forecast: {str(e)}. Could you please provide more specific details about what you'd like to forecast?",
                "type": "error"
            }
    
    async def _handle_schedule_request(self, message: str, intent: Dict, user: User, db: Session) -> Dict[str, Any]:
        """Handle forecast scheduling requests"""
        # For now, return a helpful message about scheduling
        # In a full implementation, you'd integrate with a task scheduler
        return {
            "message": "I understand you want to schedule forecasts! While I can't set up automatic scheduling yet, I can help you generate forecasts on demand. Just tell me what you'd like to forecast and I'll create it for you right away. For example, try saying 'Generate a forecast for Product A using best fit algorithm'.",
            "type": "schedule_info",
            "suggestions": [
                "Generate forecast for [product name]",
                "Predict sales for [customer name]", 
                "Forecast demand for [location name]",
                "Run best fit analysis for [item name]"
            ]
        }
    
    async def _handle_data_query(self, message: str, intent: Dict, user: User, db: Session, context: str) -> Dict[str, Any]:
        """Handle data queries and statistics"""
        try:
            # Get database statistics
            from sqlalchemy import func, distinct
            
            total_records = db.query(func.count(ForecastData.id)).scalar()
            unique_products = db.query(func.count(distinct(ForecastData.product))).scalar()
            unique_customers = db.query(func.count(distinct(ForecastData.customer))).scalar()
            unique_locations = db.query(func.count(distinct(ForecastData.location))).scalar()
            
            date_range = db.query(
                func.min(ForecastData.date),
                func.max(ForecastData.date)
            ).first()
            
            # Get recent forecasts
            recent_forecasts = db.query(SavedForecastResult).filter(
                SavedForecastResult.user_id == user.id
            ).order_by(SavedForecastResult.created_at.desc()).limit(5).all()
            
            stats_message = f"""Here's what I found in your data:

📊 **Database Overview:**
• Total records: {total_records:,}
• Date range: {date_range[0]} to {date_range[1]}
• Unique products: {unique_products}
• Unique customers: {unique_customers}  
• Unique locations: {unique_locations}

🔮 **Your Recent Forecasts:**"""
            
            if recent_forecasts:
                for forecast in recent_forecasts:
                    try:
                        config_data = json.loads(forecast.forecast_config)
                        forecast_data = json.loads(forecast.forecast_data)
                        accuracy = forecast_data.get('accuracy', 'N/A')
                        stats_message += f"\n• {forecast.name} - {accuracy}% accuracy"
                    except:
                        stats_message += f"\n• {forecast.name}"
            else:
                stats_message += "\n• No forecasts generated yet"
            
            stats_message += "\n\nWhat would you like to forecast next?"
            
            return {
                "message": stats_message,
                "type": "data_response",
                "statistics": {
                    "total_records": total_records,
                    "unique_products": unique_products,
                    "unique_customers": unique_customers,
                    "unique_locations": unique_locations,
                    "date_range": date_range
                }
            }
            
        except Exception as e:
            return {
                "message": f"I had trouble accessing your data: {str(e)}. Please make sure your database is properly connected.",
                "type": "error"
            }
    
    async def _handle_general_chat(self, message: str, context: str) -> Dict[str, Any]:
        """Handle general chat using Ollama"""
        try:
            system_prompt = f"""You are an AI assistant for a multi-variant forecasting tool. You help users understand forecasting, generate predictions, and analyze their data.

Context: {context}

You can help users with:
1. Generating forecasts by understanding natural language requests
2. Explaining forecasting concepts and algorithms
3. Providing insights about their data
4. Suggesting best practices for forecasting

When users ask about forecasting, try to guide them to be specific about:
- What they want to forecast (product, customer, location)
- Time period (weekly, monthly, yearly)
- How many periods to forecast
- Which algorithm to use (or suggest "best fit" for automatic selection)

Be helpful, concise, and focus on forecasting-related topics."""

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": f"{system_prompt}\n\nUser: {message}\nAssistant:",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get("response", "I'm sorry, I couldn't process your request.")
                
                return {
                    "message": ai_response,
                    "type": "general_response"
                }
            else:
                return {
                    "message": "I'm having trouble connecting to the AI service. How can I help you with forecasting?",
                    "type": "fallback"
                }
                
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return {
                "message": "I'm currently having technical difficulties. However, I can still help you generate forecasts! Try saying something like 'Generate a forecast for Product A using best fit algorithm' and I'll help you create it.",
                "type": "fallback"
            }
    
    def _match_entities_with_data(self, entities: Dict, products: List[str], customers: List[str], locations: List[str]) -> Dict[str, Any]:
        """Match extracted entities with actual database data"""
        matched = {
            "found_matches": False,
            "products": [],
            "customers": [],
            "locations": [],
            "suggestions": []
        }
        
        # Enhanced fuzzy matching for products
        message_words = entities.get("message_words", [])
        
        # Check if any words in the message match product names
        for product in products:
            product_words = product.lower().split()
            if any(word.lower() in product.lower() for word in message_words):
                matched["products"].append(product)
                matched["found_matches"] = True
            elif any(term.lower() in product.lower() for term in entities.get("products", [])):
                matched["products"].append(product)
                matched["found_matches"] = True
        
        # Enhanced fuzzy matching for customers  
        for customer in customers:
            if any(word.lower() in customer.lower() for word in message_words):
                matched["customers"].append(customer)
                matched["found_matches"] = True
            elif any(term.lower() in customer.lower() for term in entities.get("customers", [])):
                matched["customers"].append(customer)
                matched["found_matches"] = True
                
        # Enhanced fuzzy matching for locations
        for location in locations:
            if any(word.lower() in location.lower() for word in message_words):
                matched["locations"].append(location)
                matched["found_matches"] = True
            elif any(term.lower() in location.lower() for term in entities.get("locations", [])):
                matched["locations"].append(location)
                matched["found_matches"] = True
        
        return matched
    
    def _create_forecast_config(self, matched_entities: Dict, original_entities: Dict) -> Any:
        """Create forecast configuration from matched entities"""
        from main import ForecastConfig
        
        # Determine forecast type based on what was found
        if matched_entities["products"]:
            forecast_by = "product"
            selected_item = matched_entities["products"][0]
            "interval": None,
            "message_words": message.split()  # Add original message words for matching
            forecast_by = "customer"
            selected_item = matched_entities["customers"][0]
        elif matched_entities["locations"]:
            forecast_by = "location"
            selected_item = matched_entities["locations"][0]
        else:
            # Default fallback
            forecast_by = "product"
            selected_item = ""
        
        # Extract algorithm preference
        algorithm = original_entities.get("algorithm", "best_fit")
        
        # Extract time preferences
        interval = original_entities.get("interval", "month")
        
        # Extract periods from numbers mentioned
        periods = original_entities.get("periods", [])
        historic_period = periods[0] if len(periods) > 0 else 12
        forecast_period = periods[1] if len(periods) > 1 else 6
        
        return ForecastConfig(
            forecastBy=forecast_by,
            selectedItem=selected_item,
            algorithm=algorithm,
            interval=interval,
            historicPeriod=historic_period,
            forecastPeriod=forecast_period
        )
    
    def _generate_clarification_request(self, products: List[str], customers: List[str], locations: List[str]) -> str:
        """Generate a clarification request when entities can't be matched"""
        message = "I'd be happy to help you generate a forecast! However, I need a bit more information. "
        
        if products:
            message += f"\n\n📦 **Available Products** (showing first 10):\n"
            for i, product in enumerate(products[:10], 1):
                message += f"{i}. {product}\n"
        
        if customers:
            message += f"\n\n👥 **Available Customers** (showing first 10):\n"
            for i, customer in enumerate(customers[:10], 1):
                message += f"{i}. {customer}\n"
        
        if locations:
            message += f"\n\n📍 **Available Locations** (showing first 10):\n"
            for i, location in enumerate(locations[:10], 1):
                message += f"{i}. {location}\n"
        
        message += "\n\n💡 **Try saying something like:**\n"
        message += "• 'Generate a forecast for [specific product name]'\n"
        message += "• 'Predict sales for [customer name] using best fit'\n"
        message += "• 'Forecast monthly demand for [location name]'\n"
        message += "• 'Run a 6-month forecast for [item name]'"
        
        return message
    
    def _format_forecast_response(self, result: Any, config: Any) -> str:
        """Format forecast results into a readable chat response"""
        try:
            message = f"🎯 **Forecast Generated Successfully!**\n\n"
            message += f"**Configuration:**\n"
            message += f"• Item: {config.selectedItem}\n"
            message += f"• Algorithm: {result.selectedAlgorithm}\n"
            message += f"• Accuracy: {result.accuracy:.1f}%\n"
            message += f"• Trend: {result.trend.capitalize()}\n\n"
            
            message += f"**📈 Forecast Results:**\n"
            for i, data_point in enumerate(result.forecastData[:6]):  # Show first 6 periods
                message += f"• {data_point.period}: {data_point.quantity:.2f}\n"
            
            if len(result.forecastData) > 6:
                message += f"• ... and {len(result.forecastData) - 6} more periods\n"
            
            message += f"\n**📊 Performance Metrics:**\n"
            message += f"• Mean Absolute Error: {result.mae:.2f}\n"
            message += f"• Root Mean Square Error: {result.rmse:.2f}\n"
            
            if hasattr(result, 'allAlgorithms') and result.allAlgorithms:
                message += f"\n**🏆 Algorithm Comparison:**\n"
                sorted_algos = sorted(result.allAlgorithms, key=lambda x: x.accuracy, reverse=True)
                for i, algo in enumerate(sorted_algos[:3]):  # Top 3
                    emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    message += f"{emoji} {algo.algorithm}: {algo.accuracy:.1f}%\n"
            
            message += f"\n✅ **Forecast saved automatically!** You can view it in your saved forecasts."
            
            return message
            
        except Exception as e:
            return f"Forecast generated but I had trouble formatting the results: {str(e)}"

# Update the existing service to use the enhanced version
async def get_ai_response(context: str, message: str, user: User = None, db: Session = None) -> Dict[str, Any]:
    """Enhanced AI response with forecast generation capabilities"""
    
    if user and db:
        # Use enhanced service for authenticated users
        service = ForecastChatService()
        return await service.process_chat_message(message, user, db, context)
    else:
        # Fallback to simple chat for unauthenticated users
        try:
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
            
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": f"{context}\n\nUser: {message}\nAssistant:",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 300
                    }
                },
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "content": result.get("response", "I'm sorry, I couldn't process your request."),
                    "type": "general_response"
                }
            else:
                return {
                    "content": "I'm having trouble connecting to the AI service. Please try again later.",
                    "type": "error"
                }
                
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return {
                "content": "I'm currently having technical difficulties. Please try again later or contact support.",
                "type": "error"
            }