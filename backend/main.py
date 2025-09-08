"""
Main FastAPI application entry point - Minimal demo version.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app.core.config import get_settings
from app.core.errors import setup_error_handlers
from app.middleware.cors import setup_cors
from app.middleware.logging import LoggingMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.api.v1.api import api_router
from app.db.dependencies import connection_pool
from app.utils.logger import setup_logging, get_logger

# Setup logging first
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("Starting Citation Network Explorer API")
    
    # Initialize database connections (Neo4j and optional Redis)
    await connection_pool.initialize()
    logger.info("Database connections initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Citation Network Explorer API")
    
    # Close database connections
    await connection_pool.close_all()
    logger.info("Database connections closed")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        lifespan=lifespan,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        openapi_url="/openapi.json" if not settings.is_production else None
    )
    
    # Add security middleware
    if settings.is_production:
        # Configure TrustedHostMiddleware with trusted hosts from settings
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.trusted_hosts
        )
    
    # Add minimal middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    
    # Setup CORS
    setup_cors(app)
    
    # Setup error handlers
    setup_error_handlers(app)
    
    # Include API router
    app.include_router(api_router)
    
    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with basic information."""
        return {
            "message": "Citation Network Explorer API",
            "version": settings.app_version,
            "environment": settings.environment,
            "docs_url": "/docs" if settings.is_development else None,
            "health_check": "/api/v1/health/"
        }
    
    logger.info(f"FastAPI application created (Environment: {settings.environment})")
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
        access_log=True,
        workers=1 if settings.is_development else 4
    )