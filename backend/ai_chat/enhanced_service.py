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
        
        if intent["type"] == "data_analysis":
            return await self._handle_data_analysis(message, intent, user, db)
        elif intent["type"] == "forecast_request":
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
        
        # Data analysis keywords (for insights, not predictions)
        analysis_keywords = [
            "analysis", "analyze", "analyse", "quick analysis", "run analysis", 
            "data analysis", "insights", "summary", "overview",
            "performance", "trends", "statistics", "stats", "show data stats",
            "data stats"
        ]
        
        # Forecast generation keywords (for predictions)
        forecast_keywords = [
            "forecast", "predict", "generate forecast", "create forecast", 
            "run forecast", "prediction", "future sales", "generate", "create"
        ]
        
        # Scheduling keywords
        schedule_keywords = [
            "schedule", "set up", "automate", "recurring", "daily", 
            "weekly", "monthly", "every", "remind me"
        ]
        
        # Data query keywords
        data_keywords = [
            "show me", "what is", "how much", "total", "average", 
            "data", "records", "sales", "what are", "list", "available",
            "show products", "show customers", "show locations",
            "products", "customers", "locations", "product", "customer", "location"
        ]
        
        # Extract entities (products, customers, locations, time periods)
        entities = self._extract_entities(message)
        
        # Debug logging for intent analysis
        print(f"🔍 DEBUG - Intent analysis for: '{message}'")
        print(f"   Message lower: '{message_lower}'")
        analysis_matches = [kw for kw in analysis_keywords if kw in message_lower]
        forecast_matches = [kw for kw in forecast_keywords if kw in message_lower]
        schedule_matches = [kw for kw in schedule_keywords if kw in message_lower]
        data_matches = [kw for kw in data_keywords if kw in message_lower]
        print(f"   Analysis keyword matches: {analysis_matches}")
        print(f"   Forecast keyword matches: {forecast_matches}")
        print(f"   Schedule keyword matches: {schedule_matches}")
        print(f"   Data keyword matches: {data_matches}")
        
        # Prioritize analysis over forecasting
        if any(keyword in message_lower for keyword in analysis_keywords):
            print(f"   -> Detected as DATA_ANALYSIS")
            return {
                "type": "data_analysis",
                "entities": entities,
                "confidence": 0.9
            }
        elif any(keyword in message_lower for keyword in forecast_keywords):
            print(f"   -> Detected as FORECAST_REQUEST")
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
            print(f"   -> Detected as GENERAL_CHAT")
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
            "interval": None,
            "message_words": message.split()  # Add message words for matching
        }
        
        # Extract product identifiers (numbers, product codes, etc.)
        # Look for patterns like "Product 10009736", "product 10009736", or just "10009736"
        product_patterns = [
            r'product\s+(\w+)',  # "Product 10009736"
            r'item\s+(\w+)',     # "Item 10009736"
            r'\b(\d{6,})\b',     # Long numbers (6+ digits) likely product codes
            r'sku\s+(\w+)',      # "SKU 10009736"
            r'code\s+(\w+)'      # "Code 10009736"
        ]
        
        for pattern in product_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            entities["products"].extend(matches)
        
        # Check for generic product mentions (like "my top product", "best product", etc.)
        generic_product_patterns = [
            "top product", "best product", "main product", "primary product",
            "my product", "the product", "this product"
        ]
        
        if any(pattern in message.lower() for pattern in generic_product_patterns):
            entities["generic_product_request"] = True
        
        # Extract customer names (look for common customer patterns)
        customer_patterns = [
            r'customer\s+([A-Za-z0-9\s]+?)(?:\s|$)',
            r'client\s+([A-Za-z0-9\s]+?)(?:\s|$)'
        ]
        
        for pattern in customer_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            entities["customers"].extend([m.strip() for m in matches])
        
        # Extract location names
        location_patterns = [
            r'location\s+([A-Za-z0-9\s]+?)(?:\s|$)',
            r'site\s+([A-Za-z0-9\s]+?)(?:\s|$)',
            r'warehouse\s+([A-Za-z0-9\s]+?)(?:\s|$)'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            entities["locations"].extend([m.strip() for m in matches])
        
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
        
        # Default interval if none specified
        if not entities["interval"]:
            entities["interval"] = "month"
        
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
        
        # Default algorithm if none specified
        if not entities["algorithm"]:
            entities["algorithm"] = "best_fit"
        
        # Extract numbers for periods (only when they're actually period-related)
        # Look for patterns like "12 months", "6 periods", "next 3", "for 24"
        period_patterns = [
            r'(\d+)\s*(?:months?|periods?|weeks?|days?)',  # "12 months", "6 periods"
            r'(?:next|for)\s+(\d+)',                       # "next 12", "for 6"
            r'(\d+)\s*(?:month|period|week|day)\s*(?:forecast|prediction)',  # "12 month forecast"
        ]
        
        period_numbers = []
        for pattern in period_patterns:
            matches = re.findall(pattern, message.lower())
            period_numbers.extend([int(m) for m in matches])
        
        if period_numbers:
            entities["periods"] = period_numbers
        else:
            # Default to reasonable forecast period if none specified
            entities["periods"] = [12]  # Default 12 periods
        
        # Debug logging
        print(f"🔍 DEBUG - Extracted entities from message: '{message}'")
        print(f"   Products: {entities['products']}")
        print(f"   Algorithm: {entities['algorithm']}")
        print(f"   Interval: {entities['interval']}")
        print(f"   Periods: {entities.get('periods', [])}")
        print(f"   Generic product request: {entities.get('generic_product_request', False)}")
        
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
            
            # Debug logging
            print(f"🔍 DEBUG - Available products (first 5): {product_list[:5]}")
            print(f"🔍 DEBUG - Matched entities: {matched_entities}")
            
            # Check if it's a generic product request (regardless of other matches)
            if intent["entities"].get("generic_product_request") and not matched_entities["products"]:
                return {
                    "message": self._generate_top_products_suggestion(product_list),
                    "type": "product_suggestion",
                    "available_options": {
                        "products": product_list[:10],  # Show top 10 products
                        "customers": customer_list[:10],
                        "locations": location_list[:10]
                    }
                }
            
            # Check if we have any valid items to forecast (products, customers, or locations)
            has_valid_items = (matched_entities["products"] or 
                             matched_entities["customers"] or 
                             matched_entities["locations"])
            
            if not matched_entities["found_matches"] or not has_valid_items:
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
    
    async def _handle_data_analysis(self, message: str, intent: Dict, user: User, db: Session) -> Dict[str, Any]:
        """Handle data analysis requests"""
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
            
            # Debug logging
            print(f"🔍 DEBUG - Data Analysis - Available products (first 5): {product_list[:5]}")
            print(f"🔍 DEBUG - Data Analysis - Matched entities: {matched_entities}")
            
            # Check for general analysis requests first (highest priority)
            general_analysis_keywords = [
                "show data stats", "quick analysis", "data stats", "overall analysis",
                "general analysis", "summary", "overview", "stats", "data statistics",
                "show me my data", "my data statistics", "database stats", "database statistics"
            ]
            
            # Also check for combinations that indicate general analysis
            message_lower = message.lower()
            is_general_analysis = (
                any(keyword in message_lower for keyword in general_analysis_keywords) or
                # Check for combinations like "show me data", "my data stats", etc.
                (("show" in message_lower or "my" in message_lower) and 
                 ("data" in message_lower) and 
                 ("stats" in message_lower or "statistics" in message_lower)) or
                # Check for general requests without specific entities
                (("analysis" in message_lower or "stats" in message_lower or "statistics" in message_lower) and
                 not matched_entities["found_matches"])
            )
            
            print(f"🔍 DEBUG - General analysis check: {is_general_analysis}")
            print(f"🔍 DEBUG - Message contains: {[kw for kw in general_analysis_keywords if kw in message.lower()]}")
            
            if is_general_analysis:
                # Perform general data analysis
                analysis_result = self._perform_general_data_analysis(db)
                return {
                    "message": self._format_analysis_response(analysis_result),
                    "type": "general_analysis_completed",
                    "analysis_result": analysis_result
                }
            
            # Check if it's a generic product request (only if not general analysis)
            if intent["entities"].get("generic_product_request") and not matched_entities["products"]:
                return {
                    "message": self._generate_top_products_analysis_suggestion(product_list),
                    "type": "analysis_suggestion",
                    "available_options": {
                        "products": product_list[:10],
                        "customers": customer_list[:10],
                        "locations": location_list[:10]
                    }
                }
            
            # Check if we have any valid items to analyze (products, customers, or locations)
            has_valid_items = (matched_entities["products"] or 
                             matched_entities["customers"] or 
                             matched_entities["locations"])
            
            if not matched_entities["found_matches"] or not has_valid_items:
                # Ask for clarification
                return {
                    "message": self._generate_analysis_clarification_request(product_list, customer_list, location_list),
                    "type": "analysis_clarification_needed",
                    "available_options": {
                        "products": product_list[:10],
                        "customers": customer_list[:10],
                        "locations": location_list[:10]
                    }
                }
            
            # Perform data analysis
            analysis_result = self._perform_data_analysis(matched_entities, db)
            
            return {
                "message": self._format_analysis_response(analysis_result),
                "type": "analysis_completed",
                "analysis_result": analysis_result
            }
            
        except Exception as e:
            return {
                "message": f"I encountered an error performing the analysis: {str(e)}. Could you please provide more specific details about what you'd like to analyze?",
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
            # Import needed functions at the top
            from sqlalchemy import func, distinct
            
            message_lower = message.lower()
            
            # Check if this is a specific listing request
            if ("products" in message_lower or "product" in message_lower) and ("available" in message_lower or "list" in message_lower or "what are" in message_lower or message_lower.strip() in ["products", "product"]):
                # Handle product listing request
                products = db.query(ForecastData.product).distinct().limit(20).all()
                product_list = [p[0] for p in products if p[0]]
                
                response = "📦 **Available Products:**\n\n"
                for i, product in enumerate(product_list, 1):
                    response += f"{i}. {product}\n"
                
                if len(product_list) == 20:
                    total_products = db.query(func.count(func.distinct(ForecastData.product))).scalar()
                    response += f"\n... and {total_products - 20} more products in your database."
                
                response += "\n\n💡 **Try saying:** 'Analyze product [product_name]' or 'Generate forecast for product [product_name]'"
                
                return {
                    "message": response,
                    "type": "product_list",
                    "products": product_list
                }
            
            elif ("customers" in message_lower or "customer" in message_lower) and ("available" in message_lower or "list" in message_lower or "what are" in message_lower or message_lower.strip() in ["customers", "customer"]):
                # Handle customer listing request
                customers = db.query(ForecastData.customer).distinct().limit(20).all()
                customer_list = [c[0] for c in customers if c[0]]
                
                response = "👥 **Available Customers:**\n\n"
                for i, customer in enumerate(customer_list, 1):
                    response += f"{i}. {customer}\n"
                
                if len(customer_list) == 20:
                    total_customers = db.query(func.count(func.distinct(ForecastData.customer))).scalar()
                    response += f"\n... and {total_customers - 20} more customers in your database."
                
                response += "\n\n💡 **Try saying:** 'Analyze customer [customer_name]' or 'Generate forecast for customer [customer_name]'"
                
                return {
                    "message": response,
                    "type": "customer_list",
                    "customers": customer_list
                }
            
            elif ("locations" in message_lower or "location" in message_lower) and ("available" in message_lower or "list" in message_lower or "what are" in message_lower or message_lower.strip() in ["locations", "location"]):
                # Handle location listing request
                locations = db.query(ForecastData.location).distinct().limit(20).all()
                location_list = [l[0] for l in locations if l[0]]
                
                response = "📍 **Available Locations:**\n\n"
                for i, location in enumerate(location_list, 1):
                    response += f"{i}. {location}\n"
                
                if len(location_list) == 20:
                    total_locations = db.query(func.count(func.distinct(ForecastData.location))).scalar()
                    response += f"\n... and {total_locations - 20} more locations in your database."
                
                response += "\n\n💡 **Try saying:** 'Analyze location [location_name]' or 'Generate forecast for location [location_name]'"
                
                return {
                    "message": response,
                    "type": "location_list",
                    "locations": location_list
                }
            
            # Get database statistics
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
        extracted_products = entities.get("products", [])
        
        # First, try exact matches with extracted product identifiers
        for extracted_product in extracted_products:
            for product in products:
                if extracted_product.lower() == product.lower() or extracted_product in product:
                    matched["products"].append(product)
                    matched["found_matches"] = True
        
        # If no exact matches, try fuzzy matching with message words
        if not matched["found_matches"]:
            for product in products:
                # Check if any words in the message match product names
                if any(word.lower() in product.lower() for word in message_words):
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
        elif matched_entities["customers"]:
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
    
    def _generate_top_products_suggestion(self, products: List[str]) -> str:
        """Generate a suggestion message for top products"""
        message = "🎯 **I'd be happy to analyze your top product!** Here are your available products:\n\n"
        
        if products:
            message += "📦 **Your Products** (showing first 10):\n"
            for i, product in enumerate(products[:10], 1):
                message += f"{i}. {product}\n"
            
            message += "\n💡 **Try saying:**\n"
            message += f"• 'Generate a forecast for {products[0]}'\n"
            if len(products) > 1:
                message += f"• 'Analyze {products[1]} using best fit algorithm'\n"
            message += "• 'Run a 6-month forecast for [product name]'\n"
            message += "• 'Predict sales for [specific product]'"
        else:
            message += "❌ No products found in your database. Please upload some data first."
        
        return message
    
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
    
    def _generate_top_products_analysis_suggestion(self, products: List[str]) -> str:
        """Generate a suggestion message for top products analysis"""
        message = "📊 **I'd be happy to analyze your top product!** Here are your available products:\n\n"
        
        if products:
            message += "📦 **Your Products** (showing first 10):\n"
            for i, product in enumerate(products[:10], 1):
                message += f"{i}. {product}\n"
            
            message += "\n💡 **Try saying:**\n"
            message += f"• 'Analyze {products[0]}'\n"
            if len(products) > 1:
                message += f"• 'Run analysis on {products[1]}'\n"
            message += "• 'Show me insights for [product name]'\n"
            message += "• 'Quick analysis of [specific product]'"
        else:
            message += "❌ No products found in your database. Please upload some data first."
        
        return message
    
    def _generate_analysis_clarification_request(self, products: List[str], customers: List[str], locations: List[str]) -> str:
        """Generate a clarification request for analysis"""
        message = "📊 **I'd be happy to help you analyze your data!** However, I need a bit more information. "
        
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
        message += "• 'Analyze [specific product name]'\n"
        message += "• 'Show insights for [customer name]'\n"
        message += "• 'Quick analysis of [location name]'\n"
        message += "• 'Run analysis on [item name]'"
        
        return message
    
    def _perform_data_analysis(self, matched_entities: Dict, db: Session) -> Dict[str, Any]:
        """Perform comprehensive data analysis"""
        from sqlalchemy import func, desc
        from datetime import datetime, timedelta
        import statistics
        
        analysis_result = {
            "summary": {},
            "trends": {},
            "insights": [],
            "recommendations": []
        }
        
        try:
            # Determine what to analyze
            if matched_entities["products"]:
                target_type = "product"
                target_items = matched_entities["products"]
            elif matched_entities["customers"]:
                target_type = "customer"
                target_items = matched_entities["customers"]
            elif matched_entities["locations"]:
                target_type = "location"
                target_items = matched_entities["locations"]
            else:
                raise ValueError("No valid items to analyze")
            
            for item in target_items:
                # Get data for the item
                if target_type == "product":
                    data_query = db.query(ForecastData).filter(ForecastData.product == item)
                elif target_type == "customer":
                    data_query = db.query(ForecastData).filter(ForecastData.customer == item)
                else:  # location
                    data_query = db.query(ForecastData).filter(ForecastData.location == item)
                
                data_points = data_query.all()
                
                if not data_points:
                    continue
                
                # Basic statistics
                quantities = [dp.quantity for dp in data_points]
                total_quantity = sum(quantities)
                avg_quantity = statistics.mean(quantities)
                median_quantity = statistics.median(quantities)
                
                # Trend analysis (last 12 months vs previous 12 months)
                sorted_data = sorted(data_points, key=lambda x: x.date)
                recent_data = sorted_data[-12:] if len(sorted_data) >= 12 else sorted_data
                older_data = sorted_data[-24:-12] if len(sorted_data) >= 24 else []
                
                trend = "stable"
                trend_percentage = 0
                
                if older_data and recent_data:
                    recent_avg = statistics.mean([dp.quantity for dp in recent_data])
                    older_avg = statistics.mean([dp.quantity for dp in older_data])
                    
                    if older_avg > 0:
                        trend_percentage = ((recent_avg - older_avg) / older_avg) * 100
                        if trend_percentage > 5:
                            trend = "increasing"
                        elif trend_percentage < -5:
                            trend = "decreasing"
                
                # Seasonality detection (simple)
                monthly_averages = {}
                for dp in data_points:
                    month = dp.date.month
                    if month not in monthly_averages:
                        monthly_averages[month] = []
                    monthly_averages[month].append(dp.quantity)
                
                seasonal_pattern = {}
                for month, values in monthly_averages.items():
                    seasonal_pattern[month] = statistics.mean(values)
                
                # Peak and low months
                if seasonal_pattern:
                    peak_month = max(seasonal_pattern, key=seasonal_pattern.get)
                    low_month = min(seasonal_pattern, key=seasonal_pattern.get)
                else:
                    peak_month = low_month = None
                
                # Volatility (coefficient of variation)
                if avg_quantity > 0:
                    std_dev = statistics.stdev(quantities) if len(quantities) > 1 else 0
                    volatility = (std_dev / avg_quantity) * 100
                else:
                    volatility = 0
                
                # Store analysis results
                analysis_result["summary"][item] = {
                    "total_quantity": total_quantity,
                    "average_quantity": avg_quantity,
                    "median_quantity": median_quantity,
                    "data_points": len(data_points),
                    "date_range": {
                        "start": min(dp.date for dp in data_points).strftime("%Y-%m-%d"),
                        "end": max(dp.date for dp in data_points).strftime("%Y-%m-%d")
                    }
                }
                
                analysis_result["trends"][item] = {
                    "trend": trend,
                    "trend_percentage": trend_percentage,
                    "volatility": volatility,
                    "peak_month": peak_month,
                    "low_month": low_month,
                    "seasonal_pattern": seasonal_pattern
                }
                
                # Generate insights
                insights = []
                if trend == "increasing":
                    insights.append(f"📈 {item} shows positive growth trend (+{trend_percentage:.1f}%)")
                elif trend == "decreasing":
                    insights.append(f"📉 {item} shows declining trend ({trend_percentage:.1f}%)")
                else:
                    insights.append(f"📊 {item} shows stable performance")
                
                if volatility > 30:
                    insights.append(f"⚠️ {item} has high volatility ({volatility:.1f}%)")
                elif volatility < 10:
                    insights.append(f"✅ {item} has low volatility ({volatility:.1f}%)")
                
                if peak_month and low_month:
                    month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    insights.append(f"📅 Peak season: {month_names[peak_month]}, Low season: {month_names[low_month]}")
                
                analysis_result["insights"].extend(insights)
                
                # Generate recommendations
                recommendations = []
                if trend == "decreasing":
                    recommendations.append(f"🎯 Consider marketing campaigns for {item} to reverse declining trend")
                elif trend == "increasing":
                    recommendations.append(f"🚀 {item} is performing well - consider increasing inventory")
                
                if volatility > 30:
                    recommendations.append(f"📋 Implement demand smoothing strategies for {item}")
                
                if peak_month:
                    month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    recommendations.append(f"📦 Prepare extra inventory for {item} before {month_names[peak_month]}")
                
                analysis_result["recommendations"].extend(recommendations)
            
            return analysis_result
            
        except Exception as e:
            print(f"Error in data analysis: {e}")
            return {
                "summary": {},
                "trends": {},
                "insights": [f"❌ Error performing analysis: {str(e)}"],
                "recommendations": ["🔧 Please check your data and try again"]
            }
    
    def _format_analysis_response(self, analysis_result: Dict[str, Any]) -> str:
        """Format analysis results into a readable response"""
        try:
            message = "📊 **Data Analysis Complete!**\n\n"
            
            # Summary section
            if analysis_result["summary"]:
                message += "📈 **Summary:**\n"
                for item, summary in analysis_result["summary"].items():
                    if item == "overall":
                        # Handle general analysis format
                        message += f"**Overall Database Statistics:**\n"
                        message += f"• Total Records: {summary['total_records']:,}\n"
                        message += f"• Total Quantity: {summary['total_quantity']:,.2f}\n"
                        message += f"• Average Quantity: {summary['average_quantity']:.2f}\n"
                        message += f"• Products: {summary['unique_products']}\n"
                        message += f"• Customers: {summary['unique_customers']}\n"
                        message += f"• Locations: {summary['unique_locations']}\n"
                        message += f"• Period: {summary['date_range']['start']} to {summary['date_range']['end']}\n\n"
                        
                        # Top performers
                        if summary.get('top_products'):
                            message += f"🏆 **Top 5 Products:**\n"
                            for i, (product, qty) in enumerate(summary['top_products'], 1):
                                message += f"{i}. {product}: {qty:,.0f}\n"
                            message += "\n"
                        
                        if summary.get('top_customers'):
                            message += f"👑 **Top 5 Customers:**\n"
                            for i, (customer, qty) in enumerate(summary['top_customers'], 1):
                                message += f"{i}. {customer}: {qty:,.0f}\n"
                            message += "\n"
                        
                        if summary.get('top_locations'):
                            message += f"📍 **Top 5 Locations:**\n"
                            for i, (location, qty) in enumerate(summary['top_locations'], 1):
                                message += f"{i}. {location}: {qty:,.0f}\n"
                            message += "\n"
                    else:
                        # Handle specific item analysis format
                        message += f"**{item}:**\n"
                        message += f"• Total Quantity: {summary['total_quantity']:,.2f}\n"
                        message += f"• Average: {summary['average_quantity']:.2f}\n"
                        message += f"• Data Points: {summary['data_points']}\n"
                        message += f"• Period: {summary['date_range']['start']} to {summary['date_range']['end']}\n\n"
            
            # Trends section
            if analysis_result["trends"]:
                message += "📊 **Trends & Patterns:**\n"
                for item, trend_data in analysis_result["trends"].items():
                    if item == "overall":
                        # Handle general analysis trends
                        message += f"**Overall Business Trend:**\n"
                        if trend_data["trend"] == "increasing":
                            message += f"📈 Trend: Growing (+{trend_data['trend_percentage']:.1f}%)\n"
                        elif trend_data["trend"] == "decreasing":
                            message += f"📉 Trend: Declining ({trend_data['trend_percentage']:.1f}%)\n"
                        elif trend_data["trend"] == "stable":
                            message += f"📊 Trend: Stable\n"
                        else:
                            message += f"📊 Trend: Insufficient data for trend analysis\n"
                        message += "\n"
                    else:
                        # Handle specific item trends
                        message += f"**{item}:**\n"
                        if trend_data["trend"] == "increasing":
                            message += f"📈 Trend: Growing (+{trend_data['trend_percentage']:.1f}%)\n"
                        elif trend_data["trend"] == "decreasing":
                            message += f"📉 Trend: Declining ({trend_data['trend_percentage']:.1f}%)\n"
                        else:
                            message += f"📊 Trend: Stable\n"
                        
                        message += f"• Volatility: {trend_data['volatility']:.1f}%\n"
                        
                        if trend_data["peak_month"] and trend_data["low_month"]:
                            month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                            message += f"• Peak Month: {month_names[trend_data['peak_month']]}\n"
                            message += f"• Low Month: {month_names[trend_data['low_month']]}\n"
                        message += "\n"
            
            # Insights section
            if analysis_result["insights"]:
                message += "💡 **Key Insights:**\n"
                for insight in analysis_result["insights"]:
                    message += f"• {insight}\n"
                message += "\n"
            
            # Recommendations section
            if analysis_result["recommendations"]:
                message += "🎯 **Recommendations:**\n"
                for recommendation in analysis_result["recommendations"]:
                    message += f"• {recommendation}\n"
            
            message += "\n✨ **Need more details?** Try asking for specific metrics or time periods!"
            
            return message
            
        except Exception as e:
            return f"Analysis completed but I had trouble formatting the results: {str(e)}"
    
    def _perform_general_data_analysis(self, db: Session) -> Dict[str, Any]:
        """Perform general analysis of the entire dataset"""
        from sqlalchemy import func, desc
        from datetime import datetime, timedelta
        import statistics
        
        analysis_result = {
            "summary": {},
            "trends": {},
            "insights": [],
            "recommendations": []
        }
        
        try:
            # Get all data
            all_data = db.query(ForecastData).all()
            
            if not all_data:
                return {
                    "summary": {},
                    "trends": {},
                    "insights": ["❌ No data found in your database"],
                    "recommendations": ["📤 Please upload some data to get started"]
                }
            
            # Overall statistics
            total_records = len(all_data)
            total_quantity = sum(dp.quantity for dp in all_data)
            avg_quantity = statistics.mean([dp.quantity for dp in all_data])
            
            # Date range
            dates = [dp.date for dp in all_data]
            date_range = {
                "start": min(dates).strftime("%Y-%m-%d"),
                "end": max(dates).strftime("%Y-%m-%d")
            }
            
            # Count unique entities
            unique_products = len(set(dp.product for dp in all_data if dp.product))
            unique_customers = len(set(dp.customer for dp in all_data if dp.customer))
            unique_locations = len(set(dp.location for dp in all_data if dp.location))
            
            # Top performers
            product_totals = {}
            customer_totals = {}
            location_totals = {}
            
            for dp in all_data:
                if dp.product:
                    product_totals[dp.product] = product_totals.get(dp.product, 0) + dp.quantity
                if dp.customer:
                    customer_totals[dp.customer] = customer_totals.get(dp.customer, 0) + dp.quantity
                if dp.location:
                    location_totals[dp.location] = location_totals.get(dp.location, 0) + dp.quantity
            
            # Get top 5 in each category
            top_products = sorted(product_totals.items(), key=lambda x: x[1], reverse=True)[:5]
            top_customers = sorted(customer_totals.items(), key=lambda x: x[1], reverse=True)[:5]
            top_locations = sorted(location_totals.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Monthly trends
            monthly_data = {}
            for dp in all_data:
                month_key = dp.date.strftime("%Y-%m")
                if month_key not in monthly_data:
                    monthly_data[month_key] = []
                monthly_data[month_key].append(dp.quantity)
            
            monthly_totals = {month: sum(quantities) for month, quantities in monthly_data.items()}
            sorted_months = sorted(monthly_totals.keys())
            
            # Calculate overall trend
            if len(sorted_months) >= 6:
                recent_months = sorted_months[-3:]  # Last 3 months
                older_months = sorted_months[-6:-3]  # Previous 3 months
                
                recent_avg = statistics.mean([monthly_totals[m] for m in recent_months])
                older_avg = statistics.mean([monthly_totals[m] for m in older_months])
                
                if older_avg > 0:
                    trend_percentage = ((recent_avg - older_avg) / older_avg) * 100
                    if trend_percentage > 5:
                        overall_trend = "increasing"
                    elif trend_percentage < -5:
                        overall_trend = "decreasing"
                    else:
                        overall_trend = "stable"
                else:
                    overall_trend = "stable"
                    trend_percentage = 0
            else:
                overall_trend = "insufficient_data"
                trend_percentage = 0
            
            # Store results
            analysis_result["summary"]["overall"] = {
                "total_records": total_records,
                "total_quantity": total_quantity,
                "average_quantity": avg_quantity,
                "date_range": date_range,
                "unique_products": unique_products,
                "unique_customers": unique_customers,
                "unique_locations": unique_locations,
                "top_products": top_products,
                "top_customers": top_customers,
                "top_locations": top_locations
            }
            
            analysis_result["trends"]["overall"] = {
                "trend": overall_trend,
                "trend_percentage": trend_percentage,
                "monthly_data": monthly_totals
            }
            
            # Generate insights
            insights = []
            insights.append(f"📊 Your database contains {total_records:,} records across {unique_products} products")
            insights.append(f"🏢 Data spans {unique_customers} customers and {unique_locations} locations")
            insights.append(f"📅 Data period: {date_range['start']} to {date_range['end']}")
            
            if overall_trend == "increasing":
                insights.append(f"📈 Overall business trend is positive (+{trend_percentage:.1f}%)")
            elif overall_trend == "decreasing":
                insights.append(f"📉 Overall business trend is declining ({trend_percentage:.1f}%)")
            elif overall_trend == "stable":
                insights.append(f"📊 Overall business trend is stable")
            
            if top_products:
                insights.append(f"🏆 Top product: {top_products[0][0]} ({top_products[0][1]:,.0f} total quantity)")
            
            if top_customers:
                insights.append(f"👑 Top customer: {top_customers[0][0]} ({top_customers[0][1]:,.0f} total quantity)")
            
            if top_locations:
                insights.append(f"📍 Top location: {top_locations[0][0]} ({top_locations[0][1]:,.0f} total quantity)")
            
            analysis_result["insights"] = insights
            
            # Generate recommendations
            recommendations = []
            if overall_trend == "decreasing":
                recommendations.append("🎯 Consider analyzing declining products/customers for targeted interventions")
            elif overall_trend == "increasing":
                recommendations.append("🚀 Business is growing - consider capacity planning and inventory optimization")
            
            if unique_products > 50:
                recommendations.append("📦 Large product portfolio - consider ABC analysis to focus on key items")
            
            if unique_customers > 100:
                recommendations.append("👥 Large customer base - consider customer segmentation analysis")
            
            recommendations.append("🔍 Use specific analysis commands to dive deeper into individual products/customers/locations")
            
            analysis_result["recommendations"] = recommendations
            
            return analysis_result
            
        except Exception as e:
            print(f"Error in general data analysis: {e}")
            return {
                "summary": {},
                "trends": {},
                "insights": [f"❌ Error performing general analysis: {str(e)}"],
                "recommendations": ["🔧 Please check your data and try again"]
            }

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