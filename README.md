# Advanced Multi-variant Forecasting Tool with AI Chat

A comprehensive forecasting application with 23+ algorithms and an intelligent AI chatbot for natural language forecast generation.

## 🤖 AI Chatbot Features

### Natural Language Forecast Generation
- Generate forecasts through simple conversations
- Automatic entity extraction (products, customers, locations)
- Intelligent algorithm selection
- Auto-save generated forecasts

### Example Chat Commands
```
User: "Generate a forecast for Product A"
AI: *Creates forecast automatically*

User: "Predict monthly sales for Customer X using best fit"
AI: *Runs best fit analysis and shows results*

User: "Show me my data statistics"
AI: *Displays database overview and recent forecasts*
```

## 🚀 Quick Start

### 1. Install Ollama (for AI Chat)
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai/download
```

### 2. Start Ollama and Pull Model
```bash
ollama serve
ollama pull llama3.1  # Recommended model
```

### 3. Start the Application
```bash
# Backend
cd backend
pip install -r requirements.txt
python setup_database.py
python main.py

# Frontend
npm install
npm run dev
```

### 4. Access the Application
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- AI Chat: Click the chat button in the bottom right

## 🎯 AI Chat Capabilities

### Forecast Generation
- **Natural Language:** "Generate a forecast for Product A"
- **Algorithm Selection:** "Use random forest to predict Customer X"
- **Time Periods:** "Create a 6-month weekly forecast"
- **Best Fit:** "Run best fit analysis for Location Y"

### Data Analysis
- **Statistics:** "Show me my data overview"
- **Trends:** "What's the trend for Product A?"
- **Performance:** "How accurate are my recent forecasts?"

### Smart Features
- **Auto-completion:** Suggests available products/customers/locations
- **Error Handling:** Guides users when requests are unclear
- **Context Awareness:** Remembers conversation context
- **Voice Input:** Supports speech-to-text (browser dependent)

## 🔧 Configuration

### Environment Variables
```env
# AI Chat Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# Database Configuration  
DB_HOST=localhost
DB_PORT=5433
DB_USER=postgres
DB_PASSWORD=root
DB_NAME=forecasting_db
```

### Model Options
- **llama3.1** (Recommended) - Best balance of speed and accuracy
- **mistral** - Faster, good for quick responses
- **codellama** - Better for technical discussions
- **llama2** - Stable and well-tested

## 📊 Original Features

### 23+ Forecasting Algorithms
- Statistical methods (ARIMA, Holt-Winters, etc.)
- Machine learning (Random Forest, XGBoost, Neural Networks)
- Specialized methods (Seasonal Decomposition, Croston's)
- Auto-selection with Best Fit mode

### Advanced Capabilities
- Multi-variant forecasting
- External factor integration
- Model caching and persistence
- Real-time data fetching (FRED API)
- Comprehensive Excel exports

## 🛠️ Troubleshooting

### AI Chat Issues
1. **"AI not responding"**
   - Check if Ollama is running: `ollama serve`
   - Verify model is installed: `ollama list`
   - Check backend logs for connection errors

2. **"Slow responses"**
   - Try a smaller model: `ollama pull mistral`
   - Ensure sufficient RAM for the model
   - Consider GPU acceleration

3. **"Can't generate forecasts"**
   - Ensure you're authenticated
   - Check if data exists in database
   - Verify backend database connection

### General Issues
- **Database connection:** Ensure PostgreSQL is running
- **Backend errors:** Check Python dependencies
- **Frontend issues:** Verify Node.js version compatibility

## 🔒 Security & Privacy

- **Local AI Processing:** All AI conversations processed locally with Ollama
- **No External APIs:** Your data never leaves your infrastructure  
- **User Authentication:** Secure access with JWT tokens
- **Data Isolation:** Users can only access their own forecasts

## 📈 Performance Tips

1. **Use appropriate models** based on your hardware
2. **Enable GPU acceleration** for faster AI responses
3. **Monitor memory usage** during concurrent AI requests
4. **Cache frequently used forecasts** for better performance

---

**Need Help?** Try asking the AI chat: "How do I generate a forecast?" or "Show me examples of what I can ask you."