import os
import logging
from fastapi import Depends, HTTPException
import onnxruntime as ort
from transformers import AutoTokenizer

from core.config import settings
from utils.milvus_manager import MilvusManager

# Configure logging
logger = logging.getLogger(__name__)

# Initialize globals
tokenizer = None
ort_session = None
milvus_manager = None


def initialize_services():
    """Initialize all services on startup."""
    global tokenizer, ort_session, milvus_manager
    
    # Use CPU execution provider only
    providers = ['CPUExecutionProvider']
    
    # Log available providers
    logger.info(f"Available ONNX providers: {ort.get_available_providers()}")
    logger.info(f"Using providers: {providers}")

    try:
        # Initialize tokenizer and model
        if not os.path.exists(settings.MODEL_PATH):
            logger.error(f"Model file not found: {settings.MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {settings.MODEL_PATH}")
            
        if not os.path.exists(settings.TOKENIZER_PATH):
            logger.error(f"Tokenizer directory not found: {settings.TOKENIZER_PATH}")
            raise FileNotFoundError(f"Tokenizer directory not found: {settings.TOKENIZER_PATH}")
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(settings.TOKENIZER_PATH)
        logger.info("Tokenizer loaded successfully")

        logger.info("Loading ONNX model...")
        ort_session = ort.InferenceSession(settings.MODEL_PATH, providers=providers)
        logger.info("ONNX model loaded successfully")
        
        # Initialize Milvus
        logger.info("Initializing Milvus...")
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(settings.MILVUS_DB_PATH), exist_ok=True)
        
        milvus_manager = MilvusManager(
            db_path=settings.MILVUS_DB_PATH,
            collection_name=settings.MILVUS_COLLECTION,
            dimension=settings.EMBEDDING_DIMENSION,
            metric_type=settings.MILVUS_METRIC_TYPE,
            recreate=False  # Don't recreate the collection if it exists
        )
        logger.info("Milvus initialized successfully")
        
        # Log collection count
        try:
            count = milvus_manager.count()
            logger.info(f"Collection contains {count} embeddings")
        except Exception as e:
            logger.warning(f"Failed to get count: {str(e)}")

    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise RuntimeError(f"Failed to initialize: {str(e)}")


def close_services():
    """Clean up services on shutdown."""
    global milvus_manager
    
    try:
        if milvus_manager:
            logger.info("Closing Milvus connection...")
            milvus_manager.close()
            logger.info("Milvus connection closed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


def get_tokenizer():
    """Dependency for tokenizer."""
    if tokenizer is None:
        raise HTTPException(status_code=500, detail="Tokenizer not initialized")
    return tokenizer


def get_ort_session():
    """Dependency for ONNX Runtime session."""
    if ort_session is None:
        raise HTTPException(status_code=500, detail="ONNX model not initialized")
    return ort_session


def get_milvus_manager():
    """Dependency for Milvus Manager."""
    if milvus_manager is None:
        raise HTTPException(status_code=500, detail="Milvus not initialized")
    return milvus_manager