import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    # App Settings
    APP_NAME: str = "ruElectra Embeddings API with ONNX Runtime (CPU) and Milvus"
    DEBUG: bool = False
    
    # Model Settings
    MODEL_PATH: str = Field(default="/app/models/ruelectra_small.onnx")
    TOKENIZER_PATH: str = Field(default="/app/models/onnx_tokenizer")
    MAX_TOKEN_LENGTH: int = Field(default=512)

    # Milvus Settings
    MILVUS_DB_PATH: str = Field(default="/app/data/milvus_data.db")
    MILVUS_COLLECTION: str = Field(default="embeddings_collection")
    EMBEDDING_DIMENSION: int = Field(default=256)
    MILVUS_METRIC_TYPE: str = Field(default="COSINE")  # This is now hardcoded to COSINE only

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create a global instance
settings = Settings(
    MODEL_PATH=os.environ.get("MODEL_PATH", "/app/models/ruelectra_small.onnx"),
    TOKENIZER_PATH=os.environ.get("TOKENIZER_PATH", "/app/models/onnx_tokenizer"),
    MILVUS_DB_PATH=os.environ.get("MILVUS_DB_PATH", "/app/data/milvus_data.db"),
    MILVUS_COLLECTION=os.environ.get("MILVUS_COLLECTION", "embeddings_collection"),
    EMBEDDING_DIMENSION=int(os.environ.get("EMBEDDING_DIMENSION", "256")),
    MAX_TOKEN_LENGTH=int(os.environ.get("MAX_TOKEN_LENGTH", "512")),
    MILVUS_METRIC_TYPE="COSINE",  # Ignore env var, always use COSINE
)