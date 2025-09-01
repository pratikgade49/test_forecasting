import os
from typing import Dict, Any, List
import requests
import json

# Choose one of these implementations based on your preferred model

# Option 1: Using local Ollama (https://ollama.ai/)
async def get_ai_response(context: str, message: str) -> Dict[str, Any]:
    """Get response from local Ollama instance"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2",  # or "mistral", "mixtral", etc.
                "prompt": f"{context}\n\nUser: {message}\nAssistant:",
                "stream": False
            }
        )
        response.raise_for_status()
        result = response.json()
        return {
            "content": result["response"],
            "references": []  # Ollama doesn't provide references
        }
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return {"content": "I'm sorry, I encountered an error processing your request."}

# Option 2: Using local LM Studio server
# async def get_ai_response(context: str, message: str) -> Dict[str, Any]:
#     """Get response from LM Studio server"""
#     try:
#         response = requests.post(
#             "http://localhost:1234/v1/chat/completions",  # Default LM Studio port
#             headers={"Content-Type": "application/json"},
#             json={
#                 "messages": [
#                     {"role": "system", "content": context},
#                     {"role": "user", "content": message}
#                 ],
#                 "temperature": 0.7,
#                 "max_tokens": 500
#             }
#         )
#         response.raise_for_status()
#         result = response.json()
#         return {
#             "content": result["choices"][0]["message"]["content"],
#             "references": []
#         }
#     except Exception as e:
#         print(f"Error calling LM Studio: {e}")
#         return {"content": "I'm sorry, I encountered an error processing your request."}
