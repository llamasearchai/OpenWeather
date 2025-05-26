"""Configuration module for OpenWeather application."""
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from enum import Enum

from pydantic import (
    BaseModel, 
    Field,
    field_validator,
    model_validator,
    BaseSettings,
    HttpUrl,
    SecretStr,
    model_dump
)
from pydantic_settings import BaseSettings as PydanticBaseSettings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class Settings(PydanticBaseSettings):
    """Application settings."""
    model_config = {
        "env_file": PROJECT_ROOT / ".env",
        "env_file_encoding": "utf-8",
        "env_prefix": "OPENWEATHER_",
        "extra": "ignore"
    }

    # Core settings
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    OPENAI_API_KEY: Optional[SecretStr] = None
    HF_API_KEY: Optional[SecretStr] = None
    
    # Ollama settings
    USE_OLLAMA: bool = True
    OLLAMA_HOST: HttpUrl = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    OLLAMA_REQUEST_TIMEOUT: int = 120  # seconds
    
    # MLX settings
    USE_MLX: bool = True
    MLX_MODEL_PATH: str = "mlx-community/Mistral-7B-Instruct-v0.2-MLX"
    
    # LLM configuration
    DEFAULT_LLM_PROVIDER: str = "local_ollama"
    OPENAI_MODEL_NAME: str = "gpt-4o"
    HF_MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.2"
    LLM_TEMPERATURE: float = Field(0.3, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(2048, gt=0)
    
    # Storage paths
    SQLITE_DB_PATH: Path = Path("data/openweather.db")
    VECTOR_STORE_PATH: Path = Path("data/vector_store_chroma")
    APP_DATA_PATH: Path = Path("data/app_data")
    
    # LangChain settings
    LANGCHAIN_VERBOSE: bool = False
    LANGCHAIN_TRACING_V2: bool = False
    
    # API server configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_CORS_ORIGINS: List[str] = ["*"]
    
    # Datasette configuration
    DATASETTE_HOST: str = "127.0.0.1"
    DATASETTE_PORT: int = 8001
    
    # CLI defaults
    FORECAST_DAYS: int = Field(7, ge=1, le=16)
    
    # Stub physics model configuration
    STUB_PHYSICS_MODEL_CONFIG: Dict[str, Any] = {}
    
    @field_validator("LOG_LEVEL")
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        """Normalize log level to uppercase."""
        return v.upper()
    
    @model_validator(mode="after")
    def resolve_paths(self) -> "Settings":
        """Resolve paths relative to project root and ensure directories exist."""
        path_fields = {
            "SQLITE_DB_PATH": self.SQLITE_DB_PATH,
            "VECTOR_STORE_PATH": self.VECTOR_STORE_PATH,
            "APP_DATA_PATH": self.APP_DATA_PATH
        }
        
        for field_name, path in path_fields.items():
            resolved_path = PROJECT_ROOT / path
            if field_name == "SQLITE_DB_PATH":
                # Create parent directory for database files
                resolved_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Create directory for data stores
                resolved_path.mkdir(parents=True, exist_ok=True)
            
            # Set the resolved path back to the field
            setattr(self, field_name, resolved_path)
            
        return self

# Create global settings instance
settings = Settings()

# Ensure directories are created on import
for path in [settings.SQLITE_DB_PATH.parent, settings.VECTOR_STORE_PATH, settings.APP_DATA_PATH]:
    path.mkdir(parents=True, exist_ok=True) 