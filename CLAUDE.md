# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Intellectual Lineage Tracer** (Citation Network Explorer), an advanced academic research analysis platform that visualizes and analyzes citation networks using AI-powered insights. The project uses a modern microservices architecture with:

- **Backend**: Python FastAPI with async support, Neo4j graph database, Redis caching, Celery task queue
- **Frontend**: React 18 + TypeScript with Vite, TailwindCSS, react-force-graph for visualization
- **AI Integration**: Multiple LLM providers (OpenAI, Anthropic) via litellm for content enrichment
- **Data Sources**: OpenAlex (240M+ papers), Semantic Scholar (225M+ papers)

## Development Commands

### Backend Development

```bash
cd backend/

# Install dependencies
pip install -r requirements.txt

# Start infrastructure (Neo4j, Redis, etc.)
docker-compose -f docker-compose.dev.yml up -d

# Initialize graph database
python app/scripts/initialize_graph_system.py

# Run FastAPI server (development mode)
python main.py
# or with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run Celery worker for background tasks
celery -A app.services.graph_tasks worker --loglevel=info

# Run tests
pytest tests/ -v --cov=app

# Code quality checks
black . && isort . && flake8
```

### Frontend Development

```bash
cd frontend/

# Install dependencies
npm install

# Start development server (runs on port 3000, proxies API to localhost:8000)
npm run dev

# Build for production
npm run build

# Type checking
npm run type-check

# Linting
npm run lint
npm run lint:fix

# Run tests
npm run test
npm run test:coverage
```

## Architecture & Code Organization

### Backend Structure (`/backend`)

The backend follows a layered architecture with clear separation of concerns:

- **`app/api/v1/endpoints/`**: FastAPI route handlers organized by feature
  - `papers.py`: Paper CRUD and search operations
  - `graph.py`: Graph algorithms and network analysis
  - `openalex.py` & `semantic_scholar.py`: External data source integrations
  - `llm_enrichment.py`: AI-powered content enhancement
  - `advanced_analytics.py`: Complex analysis features

- **`app/services/`**: Business logic and external integrations
  - `graph_operations.py`: Neo4j graph database operations
  - `llm_service.py`: Unified LLM interface using litellm
  - `openalex.py` & `semantic_scholar.py`: API client implementations
  - `citation_analysis.py`: Citation network algorithms
  - Background task services (`*_tasks.py`) for Celery

- **`app/models/`**: Pydantic models for data validation
  - Separate models for each data source (OpenAlex, Semantic Scholar)
  - Common models for API responses

- **`app/db/`**: Database connection management
  - `neo4j.py`: Graph database operations
  - `redis.py`: Caching layer

- **`app/middleware/`**: Request/response processing
  - Authentication, CORS, rate limiting, metrics collection

### Frontend Structure (`/frontend`)

The frontend uses a component-based architecture with state management:

- **`src/components/`**: Reusable UI components
  - `visualization/`: Network graph components (NetworkView, CitationFlowVisualizer)
  - `modals/`: Dialog components for user interactions
  - `papers/`: Paper list and table views
  - `ui/`: Base UI components (Button, LoadingOverlay, etc.)

- **`src/services/`**: API client and external service integrations
  - `api.ts`: Axios-based API client with interceptors
  - `paperService.ts`: Paper-related API operations
  - `websocketService.ts`: Real-time updates via WebSocket

- **`src/store/`**: Zustand state management
  - `paperStore.ts`: Global paper and network state
  - `uiStore.ts`: UI state (modals, loading states)
  - `authStore.ts`: Authentication state

- **`src/hooks/`**: Custom React hooks
  - `useWebSocket.ts`: WebSocket connection management

## Key Technical Patterns

### Graph Database Operations

The system uses Neo4j for storing and analyzing citation networks. Key operations:

1. **Paper nodes**: Store paper metadata with DOI as unique identifier
2. **CITES relationships**: Directed edges representing citations
3. **Graph algorithms**: Community detection (Leiden), centrality analysis, path finding

### LLM Integration Strategy

LLM services are used for:
- Paper summarization and key insight extraction
- Citation context analysis (understanding WHY papers cite each other)
- Research theme identification

Cost management features:
- Token counting and budget limits
- Caching of generated content
- Fallback to cheaper models when budget exceeded

### Caching Architecture

Multi-tier caching strategy:
- **Redis**: API responses, LLM outputs, computed metrics
- **Frontend**: React Query for client-side caching
- **CDN**: Static assets and pre-rendered content

### Real-time Features

WebSocket integration for:
- Live collaboration on research networks
- Real-time processing status updates
- Concurrent user awareness

## Environment Configuration

### Required Environment Variables

Backend (`.env` in `/backend`):
```bash
# Neo4j Graph Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Redis Cache
REDIS_URL=redis://localhost:6379

# Optional API Keys (higher rate limits)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
SEMANTIC_SCHOLAR_API_KEY=...

# LLM Budget Controls
LLM_DAILY_BUDGET_USD=10.0
LLM_MONTHLY_BUDGET_USD=100.0
```

Frontend (`.env` in `/frontend`):
```bash
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## Testing Strategy

- **Backend**: pytest with async support, coverage target >80%
- **Frontend**: Vitest for unit tests, React Testing Library for components
- **Integration**: End-to-end tests for critical user flows
- **Performance**: Load testing for graph operations with large networks

## API Endpoints Reference

Key endpoints to understand the system:

- `GET /api/v1/papers/doi/{doi}`: Fetch paper with enrichment
- `POST /api/v1/search/papers`: Advanced search with filters
- `GET /api/v1/papers/{id}/citations?depth=2`: Multi-level citation network
- `POST /api/v1/graph/algorithms/communities`: Detect research communities
- `POST /api/v1/llm/enrich/paper/{paper_id}`: Generate AI summaries
- `WebSocket /api/v1/analytics/ws/{client_id}`: Real-time updates

## Performance Considerations

- Graph queries are optimized with indexes on DOI and paper ID
- Citation traversal uses depth limits to prevent excessive data loading
- Background tasks handle expensive operations (LLM calls, bulk imports)
- Frontend uses virtualization for large paper lists
- Network visualization progressively loads nodes for large graphs (100K+ papers)