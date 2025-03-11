from fastapi import FastAPI
import logging

from core.config import settings
from core.dependencies import initialize_services, close_services
from routers import embedding_router, milvus_collection_router, milvus_data_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title=settings.APP_NAME)

# Include routers
app.include_router(embedding_router.router)
app.include_router(milvus_collection_router.router)
app.include_router(milvus_data_router.router)


@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup."""
    logger.info("Starting application...")
    initialize_services()
    logger.info("Application started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down application...")
    close_services()
    logger.info("Application shut down successfully")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from core.dependencies import tokenizer, ort_session, milvus_manager
    
    model_status = "loaded" if ort_session is not None else "not loaded"
    milvus_status = "connected" if milvus_manager is not None else "not connected"
    
    logger.info(f"Health check: Model {model_status}, Milvus {milvus_status}")
    
    return {
        "status": "healthy", 
        "model": model_status,
        "milvus": milvus_status
    }