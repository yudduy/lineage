# Citation Network Explorer - Backend

FastAPI backend for building citation networks using OpenAlex data.

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Neo4j (via Docker)
- Redis (optional, via Docker)

### 1. Setup environment

Copy `env.example` to `.env` and edit:

```bash
cp env.example .env
```

Key settings:
- `NEO4J_PASSWORD` - Set a secure password
- `SECRET_KEY` - Random string for security  
- `OPENALEX_EMAIL` - Your email (gets higher rate limits)

### 2. Start databases

```bash
docker-compose -f docker-compose.secure.yml up -d
```

### 3. Install and run

```bash
pip install -r requirements.txt
python app/scripts/initialize_graph_system.py
python main.py
```

API: http://localhost:8000  
Docs: http://localhost:8000/docs

### 5. Test the Sync Network Builder

```bash
# Example with a DOI
curl -X POST http://localhost:8000/api/v1/openalex/network/build-sync \
  -H "Content-Type: application/json" \
  -d '{
    "identifier": "10.1038/nature12373",
    "direction": "both",
    "max_depth": 2,
    "max_per_level": 20
  }'

# Example with a paper title
curl -X POST http://localhost:8000/api/v1/openalex/network/build-sync \
  -H "Content-Type: application/json" \
  -d '{
    "identifier": "attention is all you need",
    "direction": "backward",
    "max_depth": 1,
    "max_per_level": 10
  }'
```

## 🏗️ Architecture

```
backend/
├── app/
│   ├── api/v1/              # API endpoints
│   │   ├── endpoints/       # Individual endpoint modules
│   │   └── models/          # API-specific models
│   ├── core/                # Core configuration and utilities
│   ├── db/                  # Database connections and models
│   ├── middleware/          # Custom middleware
│   ├── models/              # Pydantic models
│   ├── services/            # Business logic services
│   └── utils/               # Utility functions
├── scripts/                 # Deployment and utility scripts
├── tests/                   # Test suite
├── docker-compose.yml       # Production Docker setup
├── docker-compose.dev.yml   # Development Docker setup
└── main.py                  # Application entry point
```

## 🚀 Quick Start

### Development Setup

1. **Clone and setup environment**:
```bash
cd backend/
cp .env.example .env
# Edit .env with your configuration
```

2. **Start with Docker Compose**:
```bash
# Development environment with hot-reload
docker-compose -f docker-compose.dev.yml up -d

# Production environment
docker-compose up -d
```

3. **Access the API**:
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/v1/health/
- Neo4j Browser: http://localhost:7474
- Redis Commander: http://localhost:8081

### Manual Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

2. **Start services**:
```bash
# Start Neo4j and Redis (via Docker or locally)
docker run -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j:5.15-community
docker run -p 6379:6379 redis:7-alpine

# Start the API server
uvicorn main:app --reload --port 8000
```

3. **Start Celery workers**:
```bash
# In separate terminals
celery -A celery_worker worker --loglevel=info --concurrency=4
celery -A celery_worker beat --loglevel=info  # For scheduled tasks
```

## 📊 API Endpoints

### Core Endpoints
- `GET /api/v1/health/` - Comprehensive health check
- `GET /api/v1/health/ready` - Kubernetes readiness probe
- `GET /api/v1/health/live` - Kubernetes liveness probe
- `GET /api/v1/metrics/prometheus` - Prometheus metrics

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Refresh JWT token
- `GET /api/v1/auth/me` - Get current user info

### Papers
- `GET /api/v1/papers/{id}` - Get paper by ID
- `GET /api/v1/papers/doi/{doi}` - Get paper by DOI
- `POST /api/v1/papers/search` - Advanced paper search
- `GET /api/v1/papers/{id}/citations` - Get citation network
- `GET /api/v1/papers/{id}/related` - Get related papers

### Search
- `GET /api/v1/search/papers` - Search papers
- `GET /api/v1/search/authors` - Search authors
- `GET /api/v1/search/journals` - Search journals
- `GET /api/v1/search/suggestions` - Get search suggestions

### Zotero Integration
- `GET /api/v1/zotero/auth/login` - Start Zotero OAuth
- `GET /api/v1/zotero/collections` - Get user collections
- `POST /api/v1/zotero/import` - Import from Zotero
- `POST /api/v1/zotero/export` - Export to Zotero

### Background Tasks
- `POST /api/v1/tasks/papers/{id}/fetch-metadata` - Fetch paper metadata
- `POST /api/v1/tasks/papers/{id}/build-network` - Build citation network
- `GET /api/v1/tasks/{task_id}/status` - Get task status

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Application
DEBUG=false
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000

# Security
SECRET_KEY=your-super-secret-key-here-at-least-32-characters-long
SESSION_SECRET=your-session-secret

# Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
REDIS_URL=redis://localhost:6379/0

# External APIs
OPENALEX_EMAIL=your-email@domain.com
ZOTERO_APP_KEY=your-zotero-key
ZOTERO_APP_SECRET=your-zotero-secret

# Background Tasks
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
```

### Rate Limiting

Configure rate limits in your environment:

```bash
RATE_LIMIT_PER_MINUTE=60
API_RATE_LIMIT_PER_MINUTE=30
SEARCH_RATE_LIMIT_PER_MINUTE=10
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_papers.py

# Run with specific markers
pytest -m "not slow"
```

## 📈 Monitoring

### Health Checks
- `/api/v1/health/` - Comprehensive system health
- `/api/v1/health/ready` - Service readiness (dependencies available)  
- `/api/v1/health/live` - Service liveness (process running)

### Metrics
- Prometheus metrics at `/api/v1/metrics/prometheus`
- Custom application metrics in `/api/v1/metrics/`
- Grafana dashboards included in `grafana/dashboards/`

### Logging
- Structured JSON logging in production
- Correlation ID tracking across requests
- Performance and security event logging
- Configurable log levels and formats

## 🚀 Deployment

### Docker Production Deployment

1. **Build and deploy**:
```bash
# Build production image
docker build -t citation-network-api .

# Deploy with docker-compose
docker-compose up -d
```

2. **Scale workers**:
```bash
# Scale Celery workers
docker-compose up -d --scale worker=4
```

### Kubernetes Deployment

Kubernetes manifests available in `k8s/` directory:

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/
```

### Environment-Specific Configs

- **Development**: `docker-compose.dev.yml` - Hot reload, debug logging
- **Staging**: `docker-compose.staging.yml` - Production-like with debug features
- **Production**: `docker-compose.yml` - Optimized for performance and security

## 🔄 Migration from Express.js

This FastAPI backend is designed to be deployed alongside the existing Express.js server for gradual migration:

1. **Deploy FastAPI on different port** (e.g., 8000)
2. **Update frontend** to use new API endpoints gradually
3. **Migrate data** from existing storage to Neo4j/Redis
4. **Switch traffic** using load balancer or reverse proxy
5. **Retire Express.js** server once migration is complete

### API Compatibility
- Maintains backward compatibility where possible
- Enhanced error handling and validation
- Improved performance and scalability
- Extended functionality for new features

## 📚 Documentation

- **API Documentation**: Available at `/docs` in development mode
- **OpenAPI Spec**: Available at `/openapi.json`
- **Architecture Guide**: See `docs/architecture.md`
- **Deployment Guide**: See `docs/deployment.md`
- **Migration Guide**: See `docs/migration.md`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure tests pass: `pytest`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: Comprehensive docs at `/docs` endpoint
- **Health Checks**: Monitor service health at `/api/v1/health/`
- **Metrics**: Prometheus metrics for monitoring and alerting