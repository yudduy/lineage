"""
Application configuration management for minimal demo.
Supports environment variables and .env file configuration.
"""

import os
from typing import Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Main application settings - minimal demo version."""
    
    # Application settings
    app_name: str = Field(default="Citation Network Explorer API - Minimal Demo", env="APP_NAME")
    app_version: str = Field(default="1.0.0-demo", env="APP_VERSION")
    app_description: str = Field(
        default="Minimal demo API for citation network building via OpenAlex",
        env="APP_DESCRIPTION"
    )
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Neo4j settings (required)
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(default="password", env="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", env="NEO4J_DATABASE")
    
    # Redis settings (optional - for caching)
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    redis_enabled: bool = Field(default=False, env="REDIS_ENABLED")
    
    # OpenAlex API settings
    openalex_email: Optional[str] = Field(default=None, env="OPENALEX_EMAIL")
    openalex_rate_limit: int = Field(default=10, env="OPENALEX_RATE_LIMIT")  # requests per second
    openalex_max_retries: int = Field(default=3, env="OPENALEX_MAX_RETRIES")
    
    # Network building limits
    max_depth: int = Field(default=3, env="MAX_DEPTH")
    max_nodes_per_level: int = Field(default=50, env="MAX_NODES_PER_LEVEL")
    
    # CORS settings
    backend_cors_origins: list = Field(
        default=["http://localhost:3000", "http://localhost:5173", "https://localhost:3000"],
        env="BACKEND_CORS_ORIGINS"
    )
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="text", env="LOG_FORMAT")
    
    @validator("backend_cors_origins", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    @validator("environment")
    def validate_environment(cls, v):
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v
    
    @validator("redis_enabled", pre=True, always=True)
    def set_redis_enabled(cls, v, values):
        # Redis is enabled if REDIS_URL is provided
        if "redis_url" in values and values["redis_url"]:
            return True
        return False
    
    @validator("openalex_email")
    def validate_openalex_email(cls, v):
        if v and "@" not in v:
            raise ValueError("OPENALEX_EMAIL must be a valid email address for polite pool access")
        return v
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings