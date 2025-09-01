# AI Chatbot Setup Instructions

## Overview
The AI chatbot integration allows users to generate forecasts through natural language conversations. It uses Ollama with open-source LLMs for processing.

## Prerequisites

### 1. Install Ollama
```bash
# On macOS
brew install ollama

# On Linux
curl -fsSL https://ollama.ai/install.sh | sh

# On Windows
# Download from https://ollama.ai/download
```

### 2. Start Ollama Service
```bash
ollama serve
```

### 3. Pull a Model
```bash
# Recommended: Llama 3.1 (8B parameters - good balance of performance and speed)
ollama pull llama3.1

# Alternative options:
# ollama pull mistral        # Faster, smaller model
# ollama pull codellama      # Better for technical tasks
# ollama pull llama2         # Stable, well-tested
```

## Environment Configuration

Add these environment variables to your backend `.env` file:

```env
# Ollama Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# Optional: If running Ollama on a different host
# OLLAMA_URL=http://your-ollama-server:11434
```

## Features

### 1. Natural Language Forecast Generation
Users can generate forecasts by simply typing requests like:
- "Generate a forecast for Product A"
- "Predict monthly sales for Customer X using best fit"
- "Run a 6-month forecast for Location Y"
- "Create a forecast using random forest algorithm"

### 2. Data Queries
Users can ask about their data:
- "Show me my data statistics"
- "What products do I have?"
- "How much data is in my database?"

### 3. Intelligent Entity Extraction
The system automatically extracts:
- Product/Customer/Location names
- Time periods (weekly, monthly, yearly)
- Algorithm preferences
- Forecast periods

### 4. Auto-Save Generated Forecasts
All AI-generated forecasts are automatically saved with descriptive names.

## Testing the Integration

1. **Start the Backend:**
   ```bash
   cd backend
   python main.py
   ```

2. **Ensure Ollama is Running:**
   ```bash
   ollama list  # Should show your installed models
   ```

3. **Test the Chat:**
   - Open the frontend application
   - Click the AI chat button (bottom right)
   - Try: "Generate a forecast for my top product"

## Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running: `ollama serve`
- Check the model is installed: `ollama list`
- Verify the URL in environment variables

### Model Performance
- **Fast but less accurate:** Use `mistral` or `llama2:7b`
- **Balanced:** Use `llama3.1:8b` (recommended)
- **High accuracy:** Use `llama3.1:70b` (requires more RAM)

### Memory Requirements
- 7B models: ~8GB RAM
- 13B models: ~16GB RAM
- 70B models: ~64GB RAM

## Customization

### Change the Model
```bash
# Pull a different model
ollama pull mistral

# Update environment variable
OLLAMA_MODEL=mistral
```

### Adjust Response Style
Edit the system prompt in `enhanced_service.py` to change how the AI responds.

### Add More Intents
Extend the `_analyze_intent` method to recognize more types of requests.

## Security Notes

- The AI service runs locally with Ollama (no data sent to external APIs)
- All conversations are processed on your infrastructure
- User authentication is required for forecast generation
- Generated forecasts are saved with proper user isolation

## Performance Tips

1. **Use GPU acceleration** if available (Ollama supports CUDA)
2. **Adjust context length** based on your needs
3. **Consider model quantization** for faster inference
4. **Monitor memory usage** during concurrent requests