# 🧬 Intellectual Lineage Tracer

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5+-blue.svg)](https://typescriptlang.org)
[![Neo4j](https://img.shields.io/badge/Neo4j-5+-green.svg)](https://neo4j.com)

An advanced academic research analysis platform that transforms citation networks into intelligent research insights. Built on modern microservices architecture with LLM-powered content enrichment and advanced graph analytics.

## 🚀 Key Features

### 📊 **Multi-Source Data Integration**
- **OpenAlex**: 240M+ open scientific papers with comprehensive metadata
- **Semantic Scholar**: 225M+ papers with citation intent analysis and research embeddings  
- **LLM Enhancement**: AI-powered summaries, research themes, and contextual analysis
- **Real-time Processing**: Live updates and collaborative research exploration

### 🧠 **Advanced Analytics**
- **Intellectual Lineage Tracing**: Multi-depth citation traversal with temporal analysis
- **Research Community Detection**: Automated clustering and characterization 
- **Citation Intent Analysis**: Understanding WHY papers cite each other
- **Impact Modeling**: Influence propagation through research networks
- **Trend Forecasting**: Predictive analytics for research trajectories

### 🎯 **Interactive Visualization**
- **Force-Directed Graphs**: 2D/3D network visualization with React Force Graph
- **Community Highlighting**: Visual research cluster identification
- **Citation Flow Animation**: Real-time visualization of knowledge transfer
- **Timeline Analysis**: Research evolution over time
- **Progressive Enhancement**: Handle networks with 100K+ papers

### 🤖 **AI-Powered Intelligence**
- **Multi-Provider LLM**: OpenAI, Anthropic, and local model support
- **Content Enrichment**: Automated paper summaries and key insights
- **Research Recommendations**: AI-driven discovery of relevant papers
- **Contextual Analysis**: Intelligent citation relationship explanation

## 🏗️ Architecture Overview

### Modern Microservices Stack
- **Backend**: Python FastAPI with async/await performance
- **Frontend**: React 18 + TypeScript with Vite build system  
- **Database**: Neo4j graph database with Graph Data Science algorithms
- **Caching**: Redis multi-tier caching with intelligent optimization
- **Background Processing**: Celery distributed task processing
- **Real-time**: WebSocket integration for live collaboration

### Performance Characteristics
- **Response Time**: P95 < 500ms for cached queries, < 5s for complex analysis
- **Scalability**: 100K+ paper networks, 1000+ concurrent users
- **Cache Efficiency**: >85% hit rate with predictive warming
- **Background Processing**: 10K+ tasks per hour capacity

## 🚦 Quick Start

### Prerequisites
- **Python 3.11+** with pip
- **Node.js 18+** with npm
- **Docker & Docker Compose** for infrastructure
- **Git** for version control

### 1. Clone Repository
```bash
git clone https://github.com/your-org/intellectual-lineage-tracer.git
cd intellectual-lineage-tracer
```

### 2. Backend Setup (FastAPI Python)
```bash
cd backend/

# Install Python dependencies
pip install -r requirements.txt

# Start infrastructure services (Redis, Neo4j, Celery)
docker-compose -f docker-compose.dev.yml up -d

# Initialize graph database
python app/scripts/initialize_graph_system.py

# Start FastAPI server
python main.py
```
**Backend running at**: http://localhost:8000  
**API Documentation**: http://localhost:8000/docs

### 3. Frontend Setup (React TypeScript)
```bash
cd frontend/

# Install Node.js dependencies  
npm install

# Start development server
npm run dev
```
**Frontend running at**: http://localhost:5173

### 4. Background Processing
```bash
cd backend/

# Start Celery workers for background tasks
celery -A app.services.graph_tasks worker --loglevel=info
```

## 🎯 Usage Guide

### Basic Research Exploration

1. **Add Seed Papers**: Start with papers that define your research area
   - Enter DOI directly: `10.1000/example.doi`
   - Search by title: "machine learning interpretability"
   - Upload BibTeX files from reference managers
   - Import from Zotero collections

2. **Explore Citation Networks**: Discover connected research
   - **Backward Analysis**: Find foundational papers that influenced your seeds
   - **Forward Analysis**: Discover recent work building on your research
   - **Community Detection**: Identify research clusters and collaborations

3. **AI-Enhanced Insights**: Get intelligent research intelligence
   - **Paper Summaries**: LLM-generated abstracts and key contributions
   - **Citation Context**: Understand WHY papers cite each other
   - **Research Trends**: Identify emerging directions in your field
   - **Impact Analysis**: Measure influence and knowledge flow

### Advanced Features

#### **Intellectual Lineage Analysis**
```bash
POST /api/v1/analytics/lineage/10.1000/example.doi
{
  "depth": 5,
  "include_ai_analysis": true,
  "temporal_analysis": true
}
```

#### **Research Community Detection**
```bash
POST /api/v1/graph/algorithms/communities
{
  "algorithm": "leiden",
  "min_community_size": 10,
  "resolution": 1.0
}
```

#### **Real-time Collaboration**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/analytics/ws/client123');
ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  // Handle real-time research updates
};
```

## 🔧 Configuration

### Environment Variables

#### Backend Configuration (`.env`)
```bash
# Database Connections
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j  
NEO4J_PASSWORD=your_password
REDIS_URL=redis://localhost:6379

# API Keys (optional but recommended for higher rate limits)
OPENAI_API_KEY=sk-your_openai_key
ANTHROPIC_API_KEY=sk-ant-your_anthropic_key  
SEMANTIC_SCHOLAR_API_KEY=your_ss_key

# System Configuration
APP_ENV=development
LOG_LEVEL=INFO
RATE_LIMIT_REQUESTS_PER_MINUTE=1000

# LLM Cost Management
LLM_DAILY_BUDGET_USD=10.0
LLM_MONTHLY_BUDGET_USD=100.0
```

#### Frontend Configuration (`.env`)
```bash
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
VITE_APP_TITLE="Intellectual Lineage Tracer"
VITE_ENABLE_ANALYTICS=true
```

## 📚 API Documentation

### Core Endpoints

#### **Paper Management**
- `GET /api/v1/papers/doi/{doi}` - Retrieve paper with enrichment
- `POST /api/v1/search/papers` - Advanced paper search
- `GET /api/v1/papers/{id}/citations` - Get citation network
- `POST /api/v1/papers/batch` - Batch paper operations

#### **Graph Analytics**  
- `POST /api/v1/graph/algorithms/centrality` - Calculate influence scores
- `POST /api/v1/graph/algorithms/communities` - Detect research communities
- `POST /api/v1/graph/analysis/lineage` - Trace intellectual lineage
- `GET /api/v1/graph/metrics/{graph_name}` - Network statistics

#### **AI Enhancement**
- `POST /api/v1/llm/enrich/paper/{paper_id}` - Generate AI summaries
- `POST /api/v1/llm/analyze/citation` - Analyze citation context
- `GET /api/v1/llm/monitoring/metrics` - LLM usage analytics

#### **Real-time Features**
- `WebSocket /api/v1/analytics/ws/{client_id}` - Live updates
- `GET /api/v1/analytics/trends/detect` - Research trend detection
- `POST /api/v1/analytics/forecast/{entity_id}` - Predictive modeling

**Full API Documentation**: http://localhost:8000/docs

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