#!/usr/bin/env python3
"""
Health check utilities for AI chat services
"""

import os
import requests
from typing import Dict, Any

class AIHealthChecker:
    """Check health of AI services and dependencies"""
    
    @staticmethod
    def check_ollama_health() -> Dict[str, Any]:
        """Check if Ollama service is healthy"""
        try:
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            
            # Check if Ollama is running
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return {
                    "status": "unhealthy",
                    "message": "Ollama service is not responding",
                    "details": f"HTTP {response.status_code}"
                }
            
            # Check if required model is available
            model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
            models = response.json().get("models", [])
            model_names = [model.get("name", "").split(":")[0] for model in models]
            
            if model_name not in model_names:
                return {
                    "status": "unhealthy",
                    "message": f"Required model '{model_name}' is not available",
                    "details": f"Available models: {', '.join(model_names)}"
                }
            
            return {
                "status": "healthy",
                "message": "Ollama service is running with required model",
                "details": {
                    "url": ollama_url,
                    "model": model_name,
                    "available_models": model_names
                }
            }
            
        except requests.exceptions.ConnectionError:
            return {
                "status": "unhealthy",
                "message": "Cannot connect to Ollama service",
                "details": "Service may not be running or URL is incorrect"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": "Error checking Ollama health",
                "details": str(e)
            }
    
    @staticmethod
    def check_database_health(db) -> Dict[str, Any]:
        """Check if database connection is healthy"""
        try:
            from database import ForecastData
            
            # Simple query to test connection
            count = db.query(ForecastData).count()
            
            return {
                "status": "healthy",
                "message": "Database connection is working",
                "details": {
                    "total_records": count
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": "Database connection failed",
                "details": str(e)
            }
    
    @staticmethod
    def get_comprehensive_health() -> Dict[str, Any]:
        """Get comprehensive health check for all AI chat dependencies"""
        ollama_health = AIHealthChecker.check_ollama_health()
        
        overall_status = "healthy" if ollama_health["status"] == "healthy" else "unhealthy"
        
        return {
            "overall_status": overall_status,
            "services": {
                "ollama": ollama_health
            },
            "recommendations": AIHealthChecker._get_health_recommendations(ollama_health)
        }
    
    @staticmethod
    def _get_health_recommendations(ollama_health: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on service status"""
        recommendations = []
        
        if ollama_health["status"] == "unhealthy":
            if "not responding" in ollama_health["message"]:
                recommendations.append("Start Ollama service: ollama serve")
            elif "model" in ollama_health["message"]:
                model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
                recommendations.append(f"Pull required model: ollama pull {model_name}")
            else:
                recommendations.append("Check Ollama installation and configuration")
        
        return recommendations