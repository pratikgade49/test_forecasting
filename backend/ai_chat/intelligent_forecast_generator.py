#!/usr/bin/env python3
"""
Intelligent forecast generator that can create forecasts from natural language
"""

import re
import json
from typing import Dict, Any, Optional, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import distinct, func

from database import ForecastData, User

class IntelligentForecastGenerator:
    """Generates forecasts from natural language descriptions"""
    
    def __init__(self):
        self.algorithm_mappings = {
            "best fit": "best_fit",
            "best": "best_fit",
            "automatic": "best_fit",
            "linear": "linear_regression",
            "polynomial": "polynomial_regression",
            "exponential": "exponential_smoothing",
            "holt": "holt_winters",
            "holt winters": "holt_winters",
            "arima": "arima",
            "random forest": "random_forest",
            "forest": "random_forest",
            "seasonal": "seasonal_decomposition",
            "moving average": "moving_average",
            "average": "moving_average",
            "sarima": "sarima",
            "prophet": "prophet_like",
            "lstm": "lstm_like",
            "neural": "neural_network",
            "xgboost": "xgboost",
            "svr": "svr",
            "knn": "knn",
            "gaussian": "gaussian_process",
            "theta": "theta_method",
            "croston": "croston",
            "ses": "ses",
            "damped": "damped_trend",
            "naive": "naive_seasonal",
            "drift": "drift_method"
        }
        
        self.interval_mappings = {
            "weekly": "week",
            "week": "week",
            "monthly": "month",
            "month": "month",
            "yearly": "year",
            "year": "year",
            "annual": "year"
        }

    def can_generate_forecast(self, message: str) -> bool:
        """Check if the message is requesting forecast generation"""
        forecast_keywords = [
            "generate", "create", "forecast", "predict", "run", "analyze",
            "make", "build", "calculate", "estimate", "project"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in forecast_keywords)

    def extract_forecast_config(self, message: str, db: Session, user: User) -> Optional[Dict[str, Any]]:
        """Extract forecast configuration from natural language"""
        try:
            message_lower = message.lower()
            
            # Initialize config with defaults
            config = {
                "algorithm": "best_fit",
                "interval": "month",
                "historicPeriod": 12,
                "forecastPeriod": 6,
                "forecastBy": "product"
            }
            
            # Extract algorithm
            for keyword, algorithm in self.algorithm_mappings.items():
                if keyword in message_lower:
                    config["algorithm"] = algorithm
                    break
            
            # Extract interval
            for keyword, interval in self.interval_mappings.items():
                if keyword in message_lower:
                    config["interval"] = interval
                    break
            
            # Extract forecast period
            period_patterns = [
                r'(\d+)\s*(?:period|month|week|year)',
                r'(\d+)[-\s]*(?:period|month|week|year)',
                r'for\s+(\d+)',
                r'next\s+(\d+)'
            ]
            
            for pattern in period_patterns:
                match = re.search(pattern, message_lower)
                if match:
                    config["forecastPeriod"] = int(match.group(1))
                    break
            
            # Extract historic period
            historic_patterns = [
                r'using\s+(\d+)\s+(?:historic|historical|past)',
                r'(\d+)\s+(?:historic|historical|past)\s+(?:period|month|week|year)',
                r'last\s+(\d+)'
            ]
            
            for pattern in historic_patterns:
                match = re.search(pattern, message_lower)
                if match:
                    config["historicPeriod"] = int(match.group(1))
                    break
            
            # Extract entities (product, customer, location)
            entities = self._extract_entities(message, db)
            
            if entities:
                # Determine forecast mode based on entities
                if len(entities) == 1:
                    # Simple mode
                    entity_type, entity_value = list(entities.items())[0]
                    config["forecastBy"] = entity_type
                    config["selectedItem"] = entity_value
                elif len(entities) > 1:
                    # Advanced mode - specific combination
                    config.update(entities)
                else:
                    # Try to find top product if no specific entity mentioned
                    top_product = self._get_top_product(db)
                    if top_product:
                        config["forecastBy"] = "product"
                        config["selectedItem"] = top_product
                    else:
                        return None
            else:
                # Try to find top product if no specific entity mentioned
                top_product = self._get_top_product(db)
                if top_product:
                    config["forecastBy"] = "product"
                    config["selectedItem"] = top_product
                else:
                    return None
            
            return config
            
        except Exception as e:
            print(f"Error extracting forecast config: {e}")
            return None

    def _extract_entities(self, message: str, db: Session) -> Dict[str, str]:
        """Extract product, customer, location entities from message"""
        entities = {}
        
        try:
            # Get available options from database
            products = [p[0] for p in db.query(distinct(ForecastData.product)).filter(ForecastData.product.isnot(None)).all()]
            customers = [c[0] for c in db.query(distinct(ForecastData.customer)).filter(ForecastData.customer.isnot(None)).all()]
            locations = [l[0] for l in db.query(distinct(ForecastData.location)).filter(ForecastData.location.isnot(None)).all()]
            
            message_lower = message.lower()
            
            # Find mentioned products
            for product in products:
                if product.lower() in message_lower:
                    entities["selectedProduct"] = product
                    break
            
            # Find mentioned customers
            for customer in customers:
                if customer.lower() in message_lower:
                    entities["selectedCustomer"] = customer
                    break
            
            # Find mentioned locations
            for location in locations:
                if location.lower() in message_lower:
                    entities["selectedLocation"] = location
                    break
            
            # If only one entity found, convert to simple mode
            if len(entities) == 1:
                if "selectedProduct" in entities:
                    return {"product": entities["selectedProduct"]}
                elif "selectedCustomer" in entities:
                    return {"customer": entities["selectedCustomer"]}
                elif "selectedLocation" in entities:
                    return {"location": entities["selectedLocation"]}
            
        except Exception as e:
            print(f"Error extracting entities: {e}")
        
        return entities

    def _get_top_product(self, db: Session) -> Optional[str]:
        """Get the top product by volume"""
        try:
            top_product = db.query(
                ForecastData.product,
                func.sum(ForecastData.quantity).label('total_quantity')
            ).filter(ForecastData.product.isnot(None)).group_by(ForecastData.product).order_by(
                func.sum(ForecastData.quantity).desc()
            ).first()
            
            return top_product[0] if top_product else None
        except Exception as e:
            print(f"Error getting top product: {e}")
            return None

    def generate_forecast_description(self, config: Dict[str, Any]) -> str:
        """Generate a human-readable description of the forecast configuration"""
        try:
            algorithm_name = config.get("algorithm", "best_fit").replace("_", " ").title()
            interval = config.get("interval", "month")
            forecast_period = config.get("forecastPeriod", 6)
            historic_period = config.get("historicPeriod", 12)
            
            if config.get("selectedItem"):
                target = f"{config['forecastBy']}: {config['selectedItem']}"
            elif config.get("selectedProduct") and config.get("selectedCustomer") and config.get("selectedLocation"):
                target = f"{config['selectedProduct']} → {config['selectedCustomer']} → {config['selectedLocation']}"
            else:
                target = "selected data"
            
            return f"Generating {forecast_period}-{interval} forecast for {target} using {algorithm_name} algorithm with {historic_period} {interval}s of historical data."
            
        except Exception as e:
            return f"Generating forecast with extracted configuration: {config}"

# Global instance
intelligent_forecast_generator = IntelligentForecastGenerator()