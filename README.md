# Citation Network Explorer

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5+-blue.svg)](https://typescriptlang.org)
[![Neo4j](https://img.shields.io/badge/Neo4j-5+-green.svg)](https://neo4j.com)

A straightforward tool for exploring academic citation networks. Built with FastAPI and React, designed to be easily forked and customized for your research needs.

## What's included

### OpenAlex Integration
- Search through 240M+ academic papers 
- Rate-limited API client that won't get you banned
- Build citation networks from DOIs or paper titles

### Network Visualization  
- Interactive graphs using React Force Graph
- Click through citation networks to explore relationships
- Neo4j backend for fast graph queries

### Simple Stack
- **Backend**: FastAPI with async support
- **Frontend**: React + TypeScript 
- **Database**: Neo4j for graph data
- **Cache**: Redis (optional but recommended)

### Production Ready
- Security headers and CORS properly configured
- Rate limiting to protect against abuse
- Environment-based config (no hardcoded secrets)
- Modular architecture that's easy to extend

## Architecture

This is a standard web app with API backend and React frontend.

**Backend**: FastAPI (Python) with async support  
**Frontend**: React + TypeScript built with Vite  
**Database**: Neo4j for storing citation networks  
**Cache**: Redis for faster API responses  

### API Endpoints
- `/api/v1/health/` - Check if everything's running
- `/api/v1/papers/` - Create, read, update papers
- `/api/v1/search/` - Search for papers  
- `/api/v1/openalex/` - Build networks from OpenAlex data

## Getting Started

### What you need
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose 
- Git

### 1. Clone the repo
```bash
git clone https://github.com/your-org/citation-network-explorer.git
cd citation-network-explorer
```

### 2. Choose your setup method

**Option A: Full Docker (Recommended)**
```bash
# Copy and edit environment config
cp backend/env.example .env
# Edit .env with your passwords

# Run everything with Docker
docker-compose up -d

# Initialize the database (one time)
docker-compose exec backend python app/scripts/setup_database.py
```

**Option B: Local development**
```bash
# Start just the databases
docker-compose -f docker-compose.dev.yml up neo4j redis -d

# Run backend locally
cd backend/
pip install -r requirements.txt
cp env.example .env
python main.py

# Run frontend locally (separate terminal)
cd frontend/
npm install
npm run dev
```

**Access the app:**
- Frontend: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Neo4j Browser: http://localhost:7474

## How to use it

### Basic workflow

1. **Add some papers**: Start with DOIs or paper titles that interest you
   - Enter a DOI like `10.1000/example.doi`
   - Search by title: "attention is all you need"
   - Upload BibTeX files if you have them

2. **Explore the network**: Click around to see connections
   - See what papers cite your starting papers
   - Find papers that your starting papers cite
   - Look for clusters of related work

3. **Export your findings**: Get the data in various formats
   - BibTeX for reference managers
   - CSV for spreadsheets
   - JSON for further processing

### API Usage

If you want to build your own frontend or integrate with other tools:

```bash
# Search for papers
curl "http://localhost:8000/api/v1/search/papers?q=machine+learning"

# Get a paper by DOI
curl "http://localhost:8000/api/v1/papers/doi/10.1000/example.doi"

# Build citation network
curl -X POST "http://localhost:8000/api/v1/openalex/network/build-sync" \
  -H "Content-Type: application/json" \
  -d '{"identifier": "attention is all you need", "max_depth": 2}'
```

## Configuration

### Backend settings (backend/.env)
```bash
# Required
SECRET_KEY=your_secret_key_here
NEO4J_PASSWORD=your_neo4j_password

# Optional
OPENALEX_EMAIL=your_email@domain.com
REDIS_URL=redis://localhost:6379
```

### Frontend settings (frontend/.env)  
```bash
VITE_API_URL=http://localhost:8000/api/v1
```

The `env.example` files have all the options with comments.

## API Reference

The interactive docs are at http://localhost:8000/docs when running.

Main endpoints:
- `GET /api/v1/health/` - Check system status
- `GET /api/v1/papers/doi/{doi}` - Get paper details  
- `POST /api/v1/search/papers` - Search papers
- `POST /api/v1/openalex/network/build-sync` - Build citation networks

## 🧪 Development

### Testing
```bash
# Backend testing
cd backend/
pytest tests/ -v --cov=app

# Frontend testing  
cd frontend/
npm run test
```

### Code Quality
```bash
# Python linting and formatting
cd backend/
black . && isort . && flake8

# TypeScript linting
cd frontend/  
npm run lint && npm run type-check
```

### Database Management
```bash
# Reset Neo4j database
python backend/app/scripts/reset_database.py

# Data migration from legacy system
python backend/app/services/data_migration.py --source legacy_papers.json
```

## 🚀 Production Deployment

### Docker Deployment
```bash
# Build and start all services
docker-compose up -d

# Scale services
docker-compose up -d --scale api=3 --scale worker=5
```

### Infrastructure Requirements
- **CPU**: 4+ cores recommended  
- **RAM**: 16GB+ for large graph processing
- **Storage**: 100GB+ SSD for Neo4j database
- **Network**: 1Gbps for API data fetching

### Monitoring
- **Health Checks**: http://localhost:8000/api/v1/health/comprehensive
- **Metrics**: Prometheus endpoint at `/metrics`
- **Logs**: Structured JSON logging with correlation IDs

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test thoroughly
4. Submit pull request with detailed description

### Areas for Contribution
- **Algorithm Improvements**: Enhanced graph algorithms and community detection
- **Data Sources**: Additional academic database integrations  
- **Visualization**: New interactive features and export formats
- **Performance**: Optimization and scalability improvements
- **Documentation**: Tutorials, examples, and API documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAlex** for providing open access to scholarly data
- **Semantic Scholar** for advanced citation analysis capabilities
- **Neo4j** for powerful graph database technology
- **Original Citation Gecko** team for the foundational concept

## 📞 Support & Community

- **Issues**: [GitHub Issues](https://github.com/your-org/intellectual-lineage-tracer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/intellectual-lineage-tracer/discussions)  
- **Documentation**: [Wiki](https://github.com/your-org/intellectual-lineage-tracer/wiki)

---

**Transform your research discovery with AI-powered citation intelligence** 🚀