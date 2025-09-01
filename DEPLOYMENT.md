# Deployment Guide

## 🐳 Docker Deployment (Recommended)

### Prerequisites
- Docker and Docker Compose installed
- At least 8GB RAM (for AI models)
- 10GB free disk space

### Quick Start
```bash
# Clone and navigate to project
git clone <your-repo-url>
cd <project-directory>

# Optional: Set FRED API key for live economic data
export FRED_API_KEY=your_fred_api_key_here

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### Services Overview
- **postgres**: PostgreSQL database (port 5432)
- **ollama**: AI model service (port 11434) 
- **app**: Main application (port 8000)

### First Time Setup
1. **Wait for services to start** (may take 2-3 minutes for Ollama model download)
2. **Access application**: http://localhost:8000
3. **Default login**: username: `admin`, password: `admin123`
4. **Upload data**: Use the "Upload Data" button to add your forecasting data

### Health Checks
```bash
# Check all services
docker-compose ps

# Check application logs
docker-compose logs app

# Check Ollama model status
curl http://localhost:11434/api/tags

# Check database connection
docker-compose exec postgres psql -U postgres -d forecasting_db -c "SELECT COUNT(*) FROM forecast_data;"
```

## 🔧 Manual Installation

### 1. Database Setup
```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb forecasting_db
```

### 2. AI Service Setup
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start service and pull model
ollama serve &
ollama pull llama3.1
```

### 3. Backend Setup
```bash
cd backend
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Setup database
python setup_database.py

# Start backend
python main.py
```

### 4. Frontend Setup
```bash
npm install
npm run build  # For production
npm run dev    # For development
```

## 🌐 Production Deployment

### Environment Variables
```env
# Database
DB_HOST=your-db-host
DB_PORT=5432
DB_USER=your-db-user
DB_PASSWORD=your-secure-password
DB_NAME=forecasting_db

# AI Service
OLLAMA_URL=http://your-ollama-host:11434
OLLAMA_MODEL=llama3.1

# Security (CHANGE THESE!)
SECRET_KEY=your-super-secure-secret-key-256-bits-minimum

# Optional APIs
FRED_API_KEY=your-fred-api-key
```

### Security Considerations
1. **Change default passwords** in production
2. **Use strong SECRET_KEY** for JWT tokens
3. **Enable HTTPS** with reverse proxy (nginx/traefik)
4. **Restrict database access** to application only
5. **Monitor resource usage** (AI models are memory-intensive)

### Scaling Considerations
- **Database**: Use managed PostgreSQL for high availability
- **AI Service**: Consider GPU acceleration for faster responses
- **Application**: Use multiple replicas behind load balancer
- **Storage**: Ensure sufficient disk space for model cache

## 🔍 Troubleshooting

### Common Issues

#### "AI not responding"
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Restart Ollama service
docker-compose restart ollama

# Check model availability
docker-compose exec ollama ollama list
```

#### "Database connection failed"
```bash
# Check PostgreSQL status
docker-compose logs postgres

# Test connection
docker-compose exec postgres psql -U postgres -d forecasting_db -c "SELECT 1;"

# Reset database
docker-compose down -v
docker-compose up -d
```

#### "Backend server not responding"
```bash
# Check application logs
docker-compose logs app

# Restart application
docker-compose restart app

# Check health endpoint
curl http://localhost:8000/
```

### Performance Optimization

#### For Low-Memory Systems
```yaml
# In docker-compose.yml, use smaller model
environment:
  - OLLAMA_MODEL=mistral  # Smaller, faster model
```

#### For High-Performance Systems
```yaml
# Use larger, more accurate model
environment:
  - OLLAMA_MODEL=llama3.1:70b  # Requires ~64GB RAM
```

### Monitoring
```bash
# Monitor resource usage
docker stats

# Check service health
curl http://localhost:8000/ai/health

# View application metrics
curl http://localhost:8000/database/stats
```

## 📊 Features Verification

After deployment, verify these features work:

### ✅ Core Functionality
- [ ] User authentication (login/register)
- [ ] Data upload (Excel/CSV files)
- [ ] Forecast generation (all algorithms)
- [ ] Results visualization and export

### ✅ AI Chat Features
- [ ] Natural language forecast requests
- [ ] Data statistics queries
- [ ] Algorithm recommendations
- [ ] Auto-save generated forecasts
- [ ] Voice input (browser dependent)

### ✅ Advanced Features
- [ ] Multi-variant forecasting
- [ ] External factor integration
- [ ] Model caching and persistence
- [ ] Saved forecast management
- [ ] Live FRED data fetching

## 🚀 Next Steps

1. **Customize branding** and styling
2. **Add more AI models** for different use cases
3. **Implement forecast scheduling** with task queues
4. **Add email notifications** for completed forecasts
5. **Integrate with business intelligence tools**
6. **Add API rate limiting** for production use